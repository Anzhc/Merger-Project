import os
import torch
from safetensors.torch import load_file
from ..utils import get_params, get_device

NODE_TYPE = 'model_loading/load_model'
NODE_CATEGORY = 'Model Loading'

def execute(node, inputs):
    params = get_params(node)
    path = params.get('path')
    dtype = params.get('dtype', 'fp16')
    dtype_map = {
        'fp16': torch.float16,
        'fp32': torch.float32,
        'bf16': torch.bfloat16,
    }
    torch_dtype = dtype_map.get(dtype, torch.float16)
    device = torch.device(get_device(node))
    if not path:
        raise ValueError('No path provided for load_model')
    path = os.path.expanduser(os.path.expandvars(path))
    ext = os.path.splitext(path)[1].lower()
    if ext == '.safetensors':
        data = load_file(path, device=device)
        fmt = 'safetensors'
    else:
        data = torch.load(path, map_location=device)
        fmt = 'pt'
    if isinstance(data, dict):
        data = {
            k: (v.to(device=device, dtype=torch_dtype) if isinstance(v, torch.Tensor) else v)
            for k, v in data.items()
        }
    return {'data': data, 'format': fmt, 'dtype': dtype}


def get_spec():
    """Return UI specification for this node."""
    return {
        'type': NODE_TYPE,
        'title': 'Load Model',
        'category': 'model_loading',
        'inputs': [],
        'outputs': [{'name': 'model', 'type': 'model'}],
        'widgets': [
            {'kind': 'button', 'name': 'Choose File', 'action': '/choose_file', 'assignTo': 'path'},
            {'kind': 'text', 'name': 'File', 'bind': 'path', 'options': {'disabled': True}},
            {'kind': 'combo', 'name': 'DType', 'bind': 'dtype', 'options': {'values': ['fp32', 'fp16', 'bf16']}},
        ],
        'properties': {'path': '', 'dtype': 'fp16'},
        'tooltip': 'Load a model from disk',
    }
