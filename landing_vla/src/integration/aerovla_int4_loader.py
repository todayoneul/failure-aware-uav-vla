"""Revision-pinned, GPU-only NF4 OpenVLA + unmerged AeroVLA PEFT adapter."""
import json
import hashlib
import time
from pathlib import Path
import torch
import bitsandbytes as bnb
from peft import PeftModel
from transformers import AutoImageProcessor, AutoTokenizer, AutoModelForVision2Seq, BitsAndBytesConfig
from transformers import LogitsProcessor, LogitsProcessorList
from .projectairsim_observation_adapter import make_mosaic, make_prompt
from .projectairsim_action_adapter import parse_action, allowed_action_tokens


class ActionGrammar(LogitsProcessor):
    """Keep generation inside AeroVLA's action language.

    About one NF4 decision in seventy puts a base-OpenVLA action token (vocabulary ids
    31744-31999) where a digit belongs, e.g. `9식 49 49` for `97 49 49`. Each position is
    masked to the grammar, so the best valid token is taken; `trace` keeps what was
    rejected and how much probability the valid tokens held.
    """
    def __init__(self, tokenizer, prompt_length):
        ids=tokenizer.convert_tokens_to_ids
        self.tokenizer=tokenizer;self.start=prompt_length;self.trace=[]
        self.digits=[ids(str(d)) for d in range(10)];self.space=ids('▁');self.end=tokenizer.eos_token_id
        self.land=tokenizer('49 LAND',add_special_tokens=False)['input_ids'][-2:]
        if tokenizer.convert_ids_to_tokens(self.land)!=['▁L','AND'] or tokenizer.unk_token_id in self.digits+[self.space]:
            raise RuntimeError('Tokenizer does not split AeroVLA actions as expected')

    def __call__(self, input_ids, scores):
        allowed=allowed_action_tokens(input_ids[0,self.start:].tolist(),self.digits,self.space,self.land,self.end)
        probabilities=torch.softmax(scores[0].float(),dim=-1)
        raw=int(probabilities.argmax());chosen=max(allowed,key=lambda token:float(probabilities[token]))
        name=self.tokenizer.convert_ids_to_tokens
        self.trace.append({'position':len(self.trace),'valid_mass':float(probabilities[allowed].sum()),
                           'chosen':name(chosen),'chosen_probability':float(probabilities[chosen]),
                           'rejected':None if raw in allowed else name(raw),
                           'rejected_id':None if raw in allowed else raw,
                           'rejected_probability':None if raw in allowed else float(probabilities[raw])})
        masked=torch.full_like(scores,float('-inf'));masked[:,allowed]=scores[:,allowed]
        return masked

    def report(self):
        interventions=[item for item in self.trace if item['rejected'] is not None]
        return {'mode':'grammar','intervened':bool(interventions),'interventions':interventions,
                'minimum_valid_mass':min((item['valid_mass'] for item in self.trace),default=None)}


class AeroVLAInt4:
    def __init__(self, manifest_path, decoder='grammar'):
        if decoder not in ('grammar','free'): raise ValueError('decoder is grammar or free')
        self.decoder=decoder
        manifest=json.loads(Path(manifest_path).read_text())
        entries={item['repo']:item for item in manifest['models']}
        base=entries['openvla/openvla-7b']['snapshot']
        adapter=Path(entries['XuPeng23/AerialVLA']['snapshot'])/'aero_vla'
        self.device=torch.device('cuda:0')
        if not torch.cuda.is_available(): raise RuntimeError('GPU-only prototype requires CUDA')
        options=dict(trust_remote_code=True, local_files_only=True)
        self.tokenizer=AutoTokenizer.from_pretrained(base, **options)
        self.image_processor=AutoImageProcessor.from_pretrained(base, **options)
        quantization=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=['vision_backbone','projector','lm_head'])
        start=time.perf_counter()
        print('LOAD_BASE_START',flush=True)
        model=AutoModelForVision2Seq.from_pretrained(base, **options,
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            quantization_config=quantization, device_map={'':0},
            attn_implementation='eager')
        print('LOAD_BASE_COMPLETE',flush=True)
        before=model.get_input_embeddings().num_embeddings
        # Embedding/head are not 4-bit. Resize before injecting the LoRA adapter.
        if isinstance(model.get_output_embeddings(), bnb.nn.Linear4bit):
            raise RuntimeError('Cannot resize a quantized output head')
        model.resize_token_embeddings(len(self.tokenizer))
        print(f'RESIZE_EMBEDDINGS {before} -> {len(self.tokenizer)}',flush=True)
        self.model=PeftModel.from_pretrained(model,str(adapter),is_trainable=False,
                                              local_files_only=True)
        self.model.eval()
        # Never model.to(device), never merge_and_unload, never CPU/disk offload.
        devices=sorted({str(p.device) for p in self.model.parameters()})
        if devices != ['cuda:0']:
            raise RuntimeError(f'Unexpected parameter placement: {devices}')
        quantized=[name for name,module in self.model.named_modules() if isinstance(module,bnb.nn.Linear4bit)]
        if not quantized: raise RuntimeError('NF4 conversion did not occur')
        self.load_info={'load_seconds':time.perf_counter()-start, 'device_map':{'':0},
            'devices':devices,'linear4bit_count':len(quantized),
            'linear4bit_first':quantized[:4], 'quantization':quantization.to_dict(),
            'embeddings_before':before,'embeddings_after':len(self.tokenizer),
            'adapter_path':str(adapter), 'base_path':base,
            'parameter_dtypes':sorted({str(p.dtype) for p in self.model.parameters()}),
            'vision_dtype':str(next(model.vision_backbone.parameters()).dtype),
            'projector_dtype':str(next(model.projector.parameters()).dtype)}
        print('LOAD_ADAPTER_COMPLETE',flush=True)

    def infer(self, front_bgr, down_bgr, state, target_position, instruction, *, trace_inputs=False, reference_frames=None,
              direction_hint=True, freeform=None):
        mosaic=make_mosaic(front_bgr,down_bgr)
        prompt=make_prompt(state,target_position,instruction,direction_hint,freeform)
        inputs=self.tokenizer([prompt],return_tensors='pt',padding=True)
        pv=self.image_processor(images=mosaic,return_tensors='pt')['pixel_values']
        if list(pv.shape) != [1,6,224,224]:
            raise RuntimeError(f'Unexpected official processor output: {pv.shape}')
        inputs={k:v.to(self.device) for k,v in inputs.items()}
        inputs['pixel_values']=pv.to(self.device,dtype=torch.bfloat16)
        evidence = None
        if trace_inputs:
            actual = inputs['pixel_values'].detach().contiguous()
            digest = lambda tensor: hashlib.sha256(tensor.contiguous().view(torch.uint16).cpu().numpy().tobytes()).hexdigest()
            evidence = {'used_frame_sha256': {'front': hashlib.sha256(front_bgr.tobytes()).hexdigest(),
                                               'down': hashlib.sha256(down_bgr.tobytes()).hexdigest()},
                        'mosaic_sha256': hashlib.sha256(mosaic.tobytes()).hexdigest(),
                        'pixel_values_sha256': digest(actual), 'dtype': str(actual.dtype),
                        'device': str(actual.device), 'shape': list(actual.shape)}
            if reference_frames is not None:
                reference = self.image_processor(images=make_mosaic(*reference_frames), return_tensors='pt')['pixel_values']
                reference = reference.to(dtype=actual.dtype)
                evidence['reference_pixel_values_sha256'] = digest(reference)
                evidence['tensor_differs_from_reference'] = evidence['pixel_values_sha256'] != evidence['reference_pixel_values_sha256']
        # `free` is plain greedy decoding over the whole vocabulary, as upstream runs it.
        grammar=ActionGrammar(self.tokenizer,inputs['input_ids'].shape[-1]) if self.decoder=='grammar' else None
        options={'logits_processor':LogitsProcessorList([grammar])} if grammar else {}
        torch.cuda.synchronize()
        start=time.perf_counter()
        with torch.inference_mode():
            ids=self.model.generate(**inputs,max_new_tokens=20,do_sample=False,
                                    eos_token_id=[self.tokenizer.eos_token_id],**options)
        torch.cuda.synchronize()
        latency=(time.perf_counter()-start)*1000
        text=self.tokenizer.decode(ids[0],skip_special_tokens=False)
        try:
            parsed=parse_action(text)
            parse_error=None
        except ValueError as error:
            parsed=None
            parse_error=str(error)
        result={'prompt':prompt,'raw_output':text,'parsed_action':parsed,'parse_error':parse_error,
                'inference_ms':latency,'pixel_values_shape':list(pv.shape),
                'mosaic_size':list(mosaic.size),'input_tokens':inputs['input_ids'].shape[-1],
                'generated_tokens':ids.shape[-1]-inputs['input_ids'].shape[-1],
                'decoder':grammar.report() if grammar else {'mode':'free'}}
        if evidence is not None:
            result['input_evidence'] = evidence
        del inputs,pv,ids
        return result
