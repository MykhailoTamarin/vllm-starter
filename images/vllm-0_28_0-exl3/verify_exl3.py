import sparkinfer
trellis = [str(op) for op in sparkinfer.list_ops() if 'trellis' in str(op)]
print('trellis_moe:', 'available' if trellis else 'NOT found')
try:
    from vllm.model_executor.layers.quantization.exl3 import Exl3Config
    print('exl3.py: imported OK')
except Exception as e:
    print('exl3.py error:', e)
print('Build OK')
