import os
import torch
from safetensors.torch import save_file
from ..utils import get_params

NODE_TYPE = 'model_saving/save_model'
NODE_CATEGORY = 'Model Saving'

def execute(node, inputs):
    params = get_params(node)
    if not inputs:
        raise ValueError('Save node requires an input model')
    model_obj = inputs[0]
    data = model_obj['data'] if isinstance(model_obj, dict) else model_obj
    fmt = params.get('format', 'safetensors')
    dtype = params.get('dtype', 'fp16')
    dtype_map = {
        'fp16': torch.float16,
        'fp32': torch.float32,
        'bf16': torch.bfloat16,
    }
    torch_dtype = dtype_map.get(dtype, torch.float16)
    path = params.get('path', '.')
    path = os.path.expanduser(os.path.expandvars(path))
    name = params.get('name', 'model')
    ext = os.path.splitext(name)[1].lower()
    if fmt == 'safetensors':
        if ext != '.safetensors':
            name = os.path.splitext(name)[0] + '.safetensors'
    elif fmt == 'pt':
        if ext not in ['.pt', '.pth']:
            name = os.path.splitext(name)[0] + '.pt'
    out = os.path.join(path, name)
    if isinstance(data, dict):
        conv_data = {k: (v.to(torch_dtype) if isinstance(v, torch.Tensor) else v)
                     for k, v in data.items()}
    else:
        conv_data = data
    if fmt == 'pt':
        torch.save(conv_data, out)
    else:
        save_file(conv_data, out)
    return out


def get_spec():
    """Return UI specification for this node."""
    return {
        'type': NODE_TYPE,
        'title': 'Save Model',
        'category': 'model_saving',
        'inputs': [{'name': 'model', 'type': 'model'}],
        'outputs': [],
        'widgets': [
            {'kind': 'text', 'name': 'Name', 'bind': 'name'},
            {'kind': 'button', 'name': 'Choose Folder', 'action': '/choose_folder', 'assignTo': 'path'},
            {'kind': 'text', 'name': 'Folder', 'bind': 'path', 'options': {'disabled': True}},
            {'kind': 'combo', 'name': 'Format', 'bind': 'format', 'options': {'values': ['safetensors', 'pt']}},
            {'kind': 'combo', 'name': 'DType', 'bind': 'dtype', 'options': {'values': ['fp32', 'fp16', 'bf16']}},
        ],
        'properties': {'name': 'model', 'path': '.', 'dtype': 'fp16', 'format': 'safetensors'},
        'tooltip': 'Save the input model to disk',
    }
