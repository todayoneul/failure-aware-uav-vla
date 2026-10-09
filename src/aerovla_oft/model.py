"""AeroVLA-OFT: OpenVLA-7B (NF4) + the frozen AeroVLA adapter + a new LoRA and an OFT-style head.

Optionally a small language-vision grounding module sits between the vision encoder and the projector
(`grounding` in the configuration): FiLM on the patch features, or a cross-attention adapter in which
the instruction's tokens query the patches. Without it the model is exactly what it was.

Follows OpenVLA-OFT's released implementation: `chunk * action_dim` empty action embeddings are
appended to the prompt, one forward pass yields their final hidden states, and an MLP-ResNet head
regresses the normalised chunk with an L1 loss. The AeroVLA adapter file is only read.
"""
import json
import time
from pathlib import Path
import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel
from transformers import AutoImageProcessor, AutoModelForVision2Seq, AutoTokenizer, BitsAndBytesConfig
from src.integration.projectairsim_observation_adapter import make_mosaic
from .spec import denormalize_action, is_stop

ACTION_DIM=3
OFT_ADAPTER='oft'
AEROVLA_ADAPTER='default'


class MLPResNetBlock(nn.Module):
    def __init__(self,dim):
        super().__init__();self.ffn=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,dim),nn.ReLU())
    def forward(self,x):return x+self.ffn(x)


class ActionHead(nn.Module):
    """OpenVLA-OFT's L1 regression head: per chunk step, the `action_dim` action-token states -> one action."""
    def __init__(self,llm_dim,hidden_dim,blocks,chunk_size,bounded=False):
        super().__init__();self.chunk_size=chunk_size;self.bounded=bounded
        self.model=nn.Sequential(nn.LayerNorm(llm_dim*ACTION_DIM),nn.Linear(llm_dim*ACTION_DIM,hidden_dim),nn.ReLU(),
                                 *[MLPResNetBlock(hidden_dim) for _ in range(blocks)],nn.LayerNorm(hidden_dim),nn.Linear(hidden_dim,ACTION_DIM))
    def forward(self,states):
        # states: (batch, chunk * action_dim, llm_dim) -> (batch, chunk, action_dim)
        actions=self.model(states.reshape(states.shape[0],self.chunk_size,-1))
        # Optional: squash into the normalised range instead of leaving the clip to denormalisation.
        return torch.tanh(actions) if self.bounded else actions


class ProprioProjector(nn.Module):
    """OpenVLA-OFT's proprio projector: one extra token after the image patches."""
    def __init__(self,proprio_dim,llm_dim):
        super().__init__();self.model=nn.Sequential(nn.Linear(proprio_dim,llm_dim),nn.GELU(),nn.Linear(llm_dim,llm_dim))
    def forward(self,proprio):return self.model(proprio)


class VisualFiLM(nn.Module):
    """FiLM on the vision encoder's patch features: gamma(l) * x + beta(l), one gamma and one beta per feature channel.

    `l` is the mean of the instruction's token embeddings. The last layer starts at zero, so gamma is 1 and beta is 0 and
    the module is the identity until it has been trained. Every patch is changed in the same way."""
    def __init__(self,language_dim,visual_dim,hidden_dim):
        super().__init__();self.visual_dim=visual_dim
        self.net=nn.Sequential(nn.LayerNorm(language_dim),nn.Linear(language_dim,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,2*visual_dim))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,visual,language,mask):
        # visual (batch, patches, visual_dim); language (batch, tokens, language_dim); mask (batch, tokens), 1 for a real token.
        weights=mask.unsqueeze(-1).to(language.dtype);pooled=(language*weights).sum(1)/weights.sum(1).clamp(min=1.)
        delta=self.net(pooled);gamma=1.+delta[:,:self.visual_dim];beta=delta[:,self.visual_dim:]
        return visual*gamma.unsqueeze(1)+beta.unsqueeze(1)


class LanguageVisionAttention(nn.Module):
    """One cross-attention layer in which the instruction's tokens query the patch features.

    Each word attends over the patches (keys and values); what it reads is written back onto the patches it was read
    from, in proportion to the attention they received from it, and added to the patch features as a residual. The output
    projection starts at zero, so the module is the identity until it has been trained. Unlike FiLM, a patch is changed
    according to whether the sentence attended to it."""
    def __init__(self,language_dim,visual_dim,attention_dim,heads):
        super().__init__()
        if attention_dim%heads:raise ValueError('attention_dim must be a multiple of heads')
        self.heads=heads;self.scale=(attention_dim//heads)**-.5
        self.language_norm=nn.LayerNorm(language_dim);self.visual_norm=nn.LayerNorm(visual_dim)
        self.query=nn.Linear(language_dim,attention_dim);self.key=nn.Linear(visual_dim,attention_dim);self.value=nn.Linear(visual_dim,attention_dim)
        self.out=nn.Linear(attention_dim,visual_dim)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)

    def attention(self,visual,language,mask):
        """Attention of every word over the patches: (batch, heads, tokens, patches). Rows of padding are zero."""
        batch,tokens,_=language.shape;patches=visual.shape[1];split=lambda x,n:x.reshape(batch,n,self.heads,-1).transpose(1,2)
        seen=self.visual_norm(visual);query=split(self.query(self.language_norm(language)),tokens);key=split(self.key(seen),patches)
        weights=torch.softmax(query@key.transpose(-1,-2)*self.scale,dim=-1)
        return weights*mask[:,None,:,None].to(weights.dtype),split(self.value(seen),patches)

    def forward(self,visual,language,mask):
        weights,value=self.attention(visual,language,mask);batch,patches=visual.shape[0],visual.shape[1]
        read=weights@value                                   # (batch, heads, tokens, d): what each word found
        # Written back where it was found: a word's reading goes to each patch by the attention the patch received from that
        # word, relative to the patch it attended to most (which gets all of it). The mean over the sentence's words.
        share=weights/weights.amax(dim=-1,keepdim=True).clamp(min=1e-6)
        back=share.transpose(-1,-2)@read                     # (batch, heads, patches, d)
        back=back/mask.sum(1).clamp(min=1.).to(back.dtype)[:,None,None,None]
        return visual+self.out(back.transpose(1,2).reshape(batch,patches,-1))


def grounding_module(settings,language_dim,visual_dim):
    """The module named by a checkpoint's `grounding` settings, or None."""
    kind=(settings or {}).get('type','none')
    if kind=='none':return None
    if settings.get('position','before_projector')!='before_projector':raise ValueError('the grounding module is defined for the patch features before the projector')
    if kind=='film':return VisualFiLM(language_dim,visual_dim,settings['hidden_dim'])
    if kind=='cross_attention':return LanguageVisionAttention(language_dim,visual_dim,settings['attention_dim'],settings['heads'])
    raise ValueError(f'Unknown grounding module: {kind!r}')


def oft_prompt(instruction):
    # AeroVLA's wrapper format with the whole sentence supplied by the user: no direction word is added.
    return f'<image>\n{instruction.strip()}\nAction: '


class AeroVLAOFT:
    def __init__(self,manifest_path,config,checkpoint=None,trainable=False):
        manifest=json.loads(Path(manifest_path).read_text());entries={item['repo']:item for item in manifest['models']}
        base=entries['openvla/openvla-7b']['snapshot'];adapter=Path(entries['XuPeng23/AerialVLA']['snapshot'])/'aero_vla'
        if not torch.cuda.is_available():raise RuntimeError('CUDA is required')
        self.config=config;self.device=torch.device('cuda:0');self.chunk=config['chunk_size']
        options=dict(trust_remote_code=True,local_files_only=True)
        self.tokenizer=AutoTokenizer.from_pretrained(base,**options)
        self.image_processor=AutoImageProcessor.from_pretrained(base,**options)
        quantization=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.bfloat16,
                                        bnb_4bit_use_double_quant=True,llm_int8_skip_modules=['vision_backbone','projector','lm_head'])
        start=time.perf_counter()
        model=AutoModelForVision2Seq.from_pretrained(base,**options,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,
                                                     quantization_config=quantization,device_map={'':0},attn_implementation='eager')
        model.resize_token_embeddings(len(self.tokenizer))
        self.peft=PeftModel.from_pretrained(model,str(adapter),is_trainable=False,local_files_only=True)
        lora=config['lora']
        if checkpoint is None:
            self.peft.add_adapter(OFT_ADAPTER,LoraConfig(r=lora['rank'],lora_alpha=lora['alpha'],lora_dropout=lora['dropout'],bias='none',
                                                         target_modules=list(lora['target_modules']),init_lora_weights='gaussian'))
        else:
            self.peft.load_adapter(str(Path(checkpoint)/OFT_ADAPTER),adapter_name=OFT_ADAPTER,is_trainable=trainable,local_files_only=True)
        # Both adapters act together: AeroVLA's UAV knowledge stays, the new one learns the chunked output.
        self.peft.base_model.set_adapter([AEROVLA_ADAPTER,OFT_ADAPTER])
        self.core=self.peft.base_model.model
        llm_dim=self.core.language_model.config.hidden_size
        self.head=ActionHead(llm_dim,config['head']['hidden_dim'],config['head']['blocks'],self.chunk,
                             bounded=config['head'].get('output','linear')=='tanh').to(self.device)
        self.use_proprio=bool(config['proprio']['enabled'])
        self.proprio_projector=ProprioProjector(len(config['proprio']['fields']),llm_dim).to(self.device) if self.use_proprio else None
        projector=self.core.projector
        self.projector=projector.modules_to_save[AEROVLA_ADAPTER] if hasattr(projector,'modules_to_save') else projector
        # Language-vision grounding on the patch features, between the vision encoder and the projector (both frozen).
        self.grounding=grounding_module(config.get('grounding'),llm_dim,self.projector.fc1.in_features)
        if self.grounding is not None:self.grounding.to(self.device)
        if checkpoint is not None:
            state=torch.load(Path(checkpoint)/'head.pt',map_location=self.device)
            self.head.load_state_dict(state['head'])
            if self.use_proprio:self.proprio_projector.load_state_dict(state['proprio_projector'])
            # A checkpoint from before the module leaves it as it starts: the identity.
            if self.grounding is not None and 'grounding' in state:self.grounding.load_state_dict(state['grounding'])
        for name,parameter in self.peft.named_parameters():
            parameter.requires_grad=bool(trainable) and 'lora_' in name and f'.{OFT_ADAPTER}.' in name
        # No dropout is wanted on the frozen AeroVLA path, and the new adapter has none.
        self.peft.eval();self.head.train(bool(trainable))
        if self.grounding is not None:self.grounding.train(bool(trainable))
        self.load_info={'load_seconds':time.perf_counter()-start,'base_path':base,'aerovla_adapter':str(adapter),
                        'checkpoint':str(checkpoint) if checkpoint else None,
                        'trainable_lora':sum(p.numel() for p in self.peft.parameters() if p.requires_grad),
                        'head_parameters':sum(p.numel() for p in self.head.parameters()),
                        'grounding_parameters':sum(p.numel() for p in self.grounding.parameters()) if self.grounding is not None else 0,
                        'grounding_loaded':bool(self.grounding is not None and checkpoint is not None and 'grounding' in state)}

    def trainable_parameters(self):
        parameters=[p for p in self.peft.parameters() if p.requires_grad]+list(self.head.parameters())
        if self.use_proprio:parameters+=list(self.proprio_projector.parameters())
        return parameters+self.grounding_parameters()

    def grounding_parameters(self):
        return list(self.grounding.parameters()) if self.grounding is not None else []

    def language_tokens(self,instructions):
        """The instruction's own token embeddings (the language model's embedding table, frozen), padded: (batch, tokens, dim) and a mask.
        Only the sentence is embedded, without the prompt's wrapper, so the module is conditioned on what was asked and nothing else."""
        ids=[self.tokenizer(text.strip(),add_special_tokens=False,return_tensors='pt').input_ids[0] for text in instructions]
        length=max(len(item) for item in ids);padded=torch.full((len(ids),length),self.tokenizer.pad_token_id,dtype=torch.long);mask=torch.zeros((len(ids),length))
        for row,item in enumerate(ids):padded[row,:len(item)]=item;mask[row,:len(item)]=1.
        with torch.no_grad():embedded=self.core.get_input_embeddings()(padded.to(self.device))
        return embedded.float(),mask.to(self.device)

    def patch_tokens(self,pixel_values,instructions):
        """Image patches as the language model receives them. With a grounding module the patch features are conditioned on the
        sentence first; the vision encoder and the projector stay frozen and the projector only carries the gradient."""
        if self.grounding is None:
            with torch.no_grad():return self.projector(self.core.vision_backbone(pixel_values))
        with torch.no_grad():features=self.core.vision_backbone(pixel_values)
        language,mask=self.language_tokens(instructions)
        # The module works in float32: its changes start small and half precision would round them away inside it.
        with torch.autocast('cuda',enabled=False):conditioned=self.grounding(features.float(),language,mask)
        return self.projector(conditioned.to(features.dtype))

    def pixel_values(self,mosaics):
        values=self.image_processor(images=mosaics,return_tensors='pt')['pixel_values']
        return values.to(self.device,dtype=torch.bfloat16)

    def forward(self,pixel_values,instructions,proprio=None):
        """Normalised action chunks, shape (batch, chunk, 3), from one pass through the language model."""
        batch=len(instructions);actions=self.chunk*ACTION_DIM
        prompts=[self.tokenizer(oft_prompt(text),return_tensors='pt').input_ids[0] for text in instructions]
        length=max(len(ids) for ids in prompts)+actions+1
        input_ids=torch.full((batch,length),self.tokenizer.pad_token_id,dtype=torch.long)
        attention=torch.zeros((batch,length),dtype=torch.long);action_mask=torch.zeros((batch,length),dtype=torch.bool)
        for row,ids in enumerate(prompts):
            end=len(ids)+actions
            # Placeholder ids stand in for the action tokens; their embeddings are zeroed below. Then the stop token.
            input_ids[row,:len(ids)]=ids;input_ids[row,len(ids):end]=1;input_ids[row,end]=self.tokenizer.eos_token_id
            attention[row,:end+1]=1;action_mask[row,len(ids):end]=True
        input_ids,attention,action_mask=input_ids.to(self.device),attention.to(self.device),action_mask.to(self.device)
        patches=self.patch_tokens(pixel_values,instructions)
        with torch.no_grad():
            embeddings=self.core.get_input_embeddings()(input_ids)
            embeddings=embeddings*(~action_mask).unsqueeze(-1)
        extra=patches.shape[1]
        if self.use_proprio:
            token=self.proprio_projector(proprio.to(self.device,dtype=torch.float32)).unsqueeze(1).to(patches.dtype)
            patches=torch.cat([patches,token],dim=1);extra+=1
        sequence=torch.cat([embeddings[:,:1],patches,embeddings[:,1:]],dim=1)
        if self.head.training:sequence.requires_grad_(True)
        mask=torch.cat([attention[:,:1],torch.ones((batch,extra),dtype=attention.dtype,device=self.device),attention[:,1:]],dim=1)
        hidden=self.core.language_model.model(inputs_embeds=sequence,attention_mask=mask,use_cache=False,return_dict=True).last_hidden_state
        # Action positions shift by the inserted patch (and proprio) tokens.
        full=torch.cat([action_mask[:,:1],torch.zeros((batch,extra),dtype=torch.bool,device=self.device),action_mask[:,1:]],dim=1)
        states=hidden[full].reshape(batch,actions,-1).float()
        return self.head(states)

    def loss(self,pixel_values,instructions,targets,proprio=None):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            predicted=self.forward(pixel_values,instructions,proprio)
        return torch.nn.functional.l1_loss(predicted.float(),targets.to(self.device,dtype=torch.float32)),predicted

    def infer(self,front_bgr,down_bgr,instruction,proprio=None):
        """One decision: the whole chunk in physical units, first action first."""
        pixel_values=self.pixel_values(make_mosaic(front_bgr,down_bgr))
        vector=torch.tensor([proprio],dtype=torch.float32) if self.use_proprio else None
        torch.cuda.synchronize();start=time.perf_counter()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            normalised=self.forward(pixel_values,[instruction],vector)[0].float().cpu().tolist()
        torch.cuda.synchronize();latency=(time.perf_counter()-start)*1000
        chunk=[denormalize_action(step,self.config) for step in normalised]
        return {'prompt':oft_prompt(instruction),'chunk':chunk,'normalised_chunk':normalised,'inference_ms':latency,
                'stop':is_stop(chunk[0],self.config),'proprio':proprio if self.use_proprio else None}

    def save(self,directory,extra=None):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        self.peft.save_pretrained(str(directory),selected_adapters=[OFT_ADAPTER])
        state={'head':self.head.state_dict()}
        if self.use_proprio:state['proprio_projector']=self.proprio_projector.state_dict()
        if self.grounding is not None:state['grounding']=self.grounding.state_dict()
        torch.save(state,directory/'head.pt')
        (directory/'manifest.json').write_text(json.dumps({'config':self.config,'load_info':self.load_info,**(extra or {})},indent=1))
