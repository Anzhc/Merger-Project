import os
import torch
from safetensors.torch import load_file
from ..utils import get_params

NODE_TYPE = 'model_loading/load_model'

def execute(node, inputs):
    params = get_params(node)
    path = params.get('path')
    if not path:
        raise ValueError('No path provided for load_model')
    path = os.path.expanduser(os.path.expandvars(path))
    ext = os.path.splitext(path)[1].lower()
    if ext == '.safetensors':
        data = load_file(path)
        fmt = 'safetensors'
    else:
        data = torch.load(path, map_location='cpu')
        fmt = 'pt'
    return {'data': data, 'format': fmt}


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
        ],
        'properties': {'path': ''},
    }
