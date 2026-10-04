"""Real CUDA/NF4 probes; local random model only, no pretrained downloads."""
import argparse
import gc
import importlib.metadata
import json
import os
import platform
import subprocess
import threading
import time
import traceback
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--model-dir', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
result = {'python': platform.python_version(), 'tests': {}, 'gpu_samples': []}
stop = threading.Event()

def gpu_snapshot():
    text = subprocess.check_output([
        '/usr/lib/wsl/lib/nvidia-smi',
        '--query-gpu=name,driver_version,memory.total,memory.used,memory.free',
        '--format=csv,noheader,nounits'], text=True).strip()
    name, driver, total, used, free = [x.strip() for x in text.split(',')]
    return {'time': time.time(), 'name': name, 'driver': driver,
            'total_mib': int(total), 'used_mib': int(used), 'free_mib': int(free)}

def monitor():
    while not stop.is_set():
        try:
            result['gpu_samples'].append(gpu_snapshot())
        except Exception as exc:
            result['gpu_monitor_error'] = repr(exc)
        stop.wait(0.25)

def probe(name, operation):
    import torch
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        detail = operation()
        torch.cuda.synchronize()
        result['tests'][name] = {
            'status': 'PASS', 'seconds': time.perf_counter() - start,
            'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
            'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
            **detail}
        print(json.dumps({name: result['tests'][name]}, indent=2), flush=True)
    except Exception:
        result['tests'][name] = {'status': 'FAIL', 'traceback': traceback.format_exc()}
        raise
    finally:
        gc.collect()
        torch.cuda.empty_cache()

try:
    result['gpu_baseline'] = gpu_snapshot()
    worker = threading.Thread(target=monitor, daemon=True)
    worker.start()
    import torch
    import bitsandbytes as bnb
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, LlamaConfig, LlamaForCausalLM
    import peft
    import accelerate
    result['versions'] = {name: importlib.metadata.version(name) for name in
                          ['torch', 'torchvision', 'transformers', 'bitsandbytes', 'peft', 'accelerate']}
    result['torch_cuda'] = torch.version.cuda
    result['cuda_available'] = torch.cuda.is_available()
    assert result['cuda_available'], 'torch.cuda.is_available() is False'
    result['gpu_name'] = torch.cuda.get_device_name(0)
    result['compute_capability'] = list(torch.cuda.get_device_capability(0))
    result['torch_arch_list'] = torch.cuda.get_arch_list()
    assert result['compute_capability'] == [12, 0], result['compute_capability']
    torch.manual_seed(123)

    def bf16():
        x = torch.randn(256, 256, device='cuda', dtype=torch.bfloat16) * 0.05
        y = torch.randn(256, 256, device='cuda', dtype=torch.bfloat16) * 0.05
        actual = x @ y
        reference = x.float() @ y.float()
        relative = ((actual.float() - reference).norm() / reference.norm()).item()
        assert torch.isfinite(actual).all().item()
        assert actual.dtype == torch.bfloat16 and relative < 0.02, relative
        return {'dtype': str(actual.dtype), 'relative_l2_error_vs_fp32': relative,
                'shape': list(actual.shape), 'bf16_supported': torch.cuda.is_bf16_supported()}
    probe('bf16_matmul', bf16)

    def nf4():
        weights = torch.randn(128, 512, dtype=torch.bfloat16)
        layer = bnb.nn.Linear4bit(512, 128, bias=False, compute_dtype=torch.bfloat16,
                                 compress_statistics=True, quant_type='nf4')
        layer.weight = bnb.nn.Params4bit(weights, requires_grad=False,
                                         compress_statistics=True, quant_type='nf4')
        layer = layer.to('cuda')
        x = torch.randn(4, 512, device='cuda', dtype=torch.bfloat16)
        actual = layer(x)
        dense_quantized = bnb.functional.dequantize_4bit(layer.weight.data, layer.weight.quant_state)
        reference = x @ dense_quantized.T
        relative = ((actual.float() - reference.float()).norm() / reference.float().norm()).item()
        assert isinstance(layer.weight, bnb.nn.Params4bit) and layer.weight.quant_state is not None
        assert torch.isfinite(actual).all().item() and relative < 0.03, relative
        return {'quant_type': layer.weight.quant_state.quant_type,
                'storage_dtype': str(layer.weight.dtype), 'output_dtype': str(actual.dtype),
                'relative_l2_error_vs_dequantized_matmul': relative,
                'shape': list(actual.shape)}
    probe('bnb_nf4_linear', nf4)

    def hf_load():
        config = LlamaConfig(vocab_size=256, hidden_size=128, intermediate_size=256,
                             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                             max_position_embeddings=64, bos_token_id=1, eos_token_id=2, pad_token_id=0)
        source = LlamaForCausalLM(config).to(dtype=torch.bfloat16)
        parameters = sum(p.numel() for p in source.parameters())
        source.save_pretrained(args.model_dir, safe_serialization=True)
        del source
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                  bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir, quantization_config=quant, torch_dtype=torch.bfloat16,
            device_map={'': 0}, low_cpu_mem_usage=True, attn_implementation='eager',
            local_files_only=True)
        modules = [name for name, module in model.named_modules() if isinstance(module, bnb.nn.Linear4bit)]
        assert model.is_loaded_in_4bit and modules, 'HF did not convert to Linear4bit'
        ids = torch.tensor([[1, 4, 7, 9, 11, 15, 21, 33]], device='cuda')
        with torch.inference_mode():
            logits = model(input_ids=ids).logits
            output = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=4,
                                    do_sample=False, pad_token_id=0)
        assert torch.isfinite(logits).all().item() and output.shape[1] > ids.shape[1]
        return {'parameters': parameters, 'checkpoint_bytes': sum(p.stat().st_size for p in args.model_dir.iterdir()),
                'source': 'locally generated random Llama; no HF pretrained checkpoint downloaded',
                'quantization_config': quant.to_dict(), 'linear4bit_modules': modules,
                'device_map': {k: str(v) for k, v in model.hf_device_map.items()},
                'logits_shape': list(logits.shape), 'generated_ids': output.cpu().tolist()}
    probe('huggingface_nf4_load_and_generate', hf_load)
    result['gate1'] = 'PASS'
except Exception:
    result['gate1'] = 'FAIL'
    result['full_traceback'] = traceback.format_exc()
    print(result['full_traceback'], flush=True)
finally:
    stop.set()
    if 'worker' in globals():
        worker.join(timeout=2)
    if result['gpu_samples']:
        result['sampled_global_gpu_peak_mib'] = max(s['used_mib'] for s in result['gpu_samples'])
    args.output.joinpath('gate1-result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'gate1': result.get('gate1'), 'result': str(args.output / 'gate1-result.json')}, indent=2))

raise SystemExit(0 if result.get('gate1') == 'PASS' else 1)
