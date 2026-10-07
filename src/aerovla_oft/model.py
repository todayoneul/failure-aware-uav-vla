"""AeroVLA-OFT: OpenVLA-7B (NF4) + the frozen AeroVLA adapter + a new LoRA and an OFT-style head.

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
    def __init__(self,llm_dim,hidden_dim,blocks,chunk_size):
        super().__init__();self.chunk_size=chunk_size
        self.model=nn.Sequential(nn.LayerNorm(llm_dim*ACTION_DIM),nn.Linear(llm_dim*ACTION_DIM,hidden_dim),nn.ReLU(),
                                 *[MLPResNetBlock(hidden_dim) for _ in range(blocks)],nn.LayerNorm(hidden_dim),nn.Linear(hidden_dim,ACTION_DIM))
    def forward(self,states):
        # states: (batch, chunk * action_dim, llm_dim) -> (batch, chunk, action_dim)
        return self.model(states.reshape(states.shape[0],self.chunk_size,-1))


class ProprioProjector(nn.Module):
    """OpenVLA-OFT's proprio projector: one extra token after the image patches."""
    def __init__(self,proprio_dim,llm_dim):
        super().__init__();self.model=nn.Sequential(nn.Linear(proprio_dim,llm_dim),nn.GELU(),nn.Linear(llm_dim,llm_dim))
    def forward(self,proprio):return self.model(proprio)


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
        self.head=ActionHead(llm_dim,config['head']['hidden_dim'],config['head']['blocks'],self.chunk).to(self.device)
        self.use_proprio=bool(config['proprio']['enabled'])
        self.proprio_projector=ProprioProjector(len(config['proprio']['fields']),llm_dim).to(self.device) if self.use_proprio else None
        if checkpoint is not None:
            state=torch.load(Path(checkpoint)/'head.pt',map_location=self.device)
            self.head.load_state_dict(state['head'])
            if self.use_proprio:self.proprio_projector.load_state_dict(state['proprio_projector'])
        for name,parameter in self.peft.named_parameters():
            parameter.requires_grad=bool(trainable) and 'lora_' in name and f'.{OFT_ADAPTER}.' in name
        # No dropout is wanted on the frozen AeroVLA path, and the new adapter has none.
        self.peft.eval();self.head.train(bool(trainable))
        projector=self.core.projector
        self.projector=projector.modules_to_save[AEROVLA_ADAPTER] if hasattr(projector,'modules_to_save') else projector
        self.load_info={'load_seconds':time.perf_counter()-start,'base_path':base,'aerovla_adapter':str(adapter),
                        'checkpoint':str(checkpoint) if checkpoint else None,
                        'trainable_lora':sum(p.numel() for p in self.peft.parameters() if p.requires_grad),
                        'head_parameters':sum(p.numel() for p in self.head.parameters())}

    def trainable_parameters(self):
        parameters=[p for p in self.peft.parameters() if p.requires_grad]+list(self.head.parameters())
        if self.use_proprio:parameters+=list(self.proprio_projector.parameters())
        return parameters

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
        with torch.no_grad():
            patches=self.projector(self.core.vision_backbone(pixel_values))
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
        torch.save(state,directory/'head.pt')
        (directory/'manifest.json').write_text(json.dumps({'config':self.config,'load_info':self.load_info,**(extra or {})},indent=1))
