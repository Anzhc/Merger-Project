from ..utils import get_params
from utils.peft_utils import parse_weights, merge_state_dicts, dare_linear
# Inspired by HuggingFace PEFT implementations

NODE_TYPE = 'peft_merging/dare_linear'
NODE_CATEGORY = 'PEFT merging'


def execute(node, inputs):
    params = get_params(node)
    if len(inputs) < 2:
        raise ValueError('DARE linear merge requires at least two inputs')

    weights = parse_weights(params.get('weights', ''), len(inputs))
    density = float(params.get('density', 0.5))

    models = []
    dtype = None
    fmt = 'pt'
    for inp in inputs:
        if isinstance(inp, dict):
            models.append(inp.get('data'))
            if dtype is None:
                dtype = inp.get('dtype')
            if fmt == 'pt':
                fmt = inp.get('format', 'pt')
        else:
            models.append(inp)

    merged = merge_state_dicts(models, weights, dare_linear, density=density)
    return {'data': merged, 'format': fmt, 'dtype': dtype}


def get_spec():
    return {
        'type': NODE_TYPE,
        'title': 'DARE Linear Merge',
        'category': 'peft_merging',
        'inputs': [
            {'name': 'A', 'type': 'model'},
            {'name': 'B', 'type': 'model'},
            {'name': 'C', 'type': 'model'},
            {'name': 'D', 'type': 'model'},
            {'name': 'E', 'type': 'model'},
            {'name': 'F', 'type': 'model'},
            {'name': 'G', 'type': 'model'},
            {'name': 'H', 'type': 'model'},
            {'name': 'I', 'type': 'model'},
            {'name': 'J', 'type': 'model'},
        ],
        'outputs': [{'name': 'model', 'type': 'model'}],
        'widgets': [
            {'kind': 'text', 'name': 'Weights (comma separated)', 'bind': 'weights'},
            {'kind': 'slider', 'name': 'Density', 'bind': 'density', 'options': {'min': 0, 'max': 1, 'step': 0.01}},
        ],
        'properties': {'weights': '', 'density': 0.5},
        'tooltip': 'Combine PEFT tensors with DARE linear algorithm (https://arxiv.org/pdf/2311.03099v3).\nDensity controls random pruning before merging.'
    }
