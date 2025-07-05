import os
import torch
from safetensors.torch import save_file
from ..utils import get_params

NODE_TYPE = 'model_saving/save_model'

def execute(node, inputs):
    params = get_params(node)
    if not inputs:
        raise ValueError('Save node requires an input model')
    model_obj = inputs[0]
    data = model_obj['data'] if isinstance(model_obj, dict) else model_obj
    fmt = model_obj.get('format', 'pt') if isinstance(model_obj, dict) else 'pt'
    path = params.get('path', '.')
    path = os.path.expanduser(os.path.expandvars(path))
    name = params.get('name', 'model')
    ext = os.path.splitext(name)[1].lower()
    if fmt == 'safetensors':
        if ext != '.safetensors':
            name = os.path.splitext(name)[0] + '.safetensors'
    else:
        if ext not in ['.pt', '.pth']:
            name = os.path.splitext(name)[0] + '.pt'
    out = os.path.join(path, name)
    if fmt == 'safetensors':
        save_file(data, out)
    else:
        torch.save(data, out)
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
        ],
        'properties': {'name': 'model', 'path': '.'},
    }
