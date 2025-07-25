from ..utils import get_params
from utils.peft_utils import merge_state_dicts, ties
# Inspired by HuggingFace PEFT implementations

NODE_TYPE = 'peft_merging/ties'
NODE_CATEGORY = 'PEFT merging'


def execute(node, inputs):
    params = get_params(node)
    if len(inputs) < 2:
        raise ValueError('TIES merge requires at least two inputs')

    dropout = float(params.get('dropout', 0.0))
    density = 1.0 - dropout
    majority = params.get('majority_sign_method', 'total')

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

    weights = [1.0] * len(models)
    merged = merge_state_dicts(models, weights, ties, density=density, majority_sign_method=majority)
    return {'data': merged, 'format': fmt, 'dtype': dtype}


def get_spec():
    return {
        'type': NODE_TYPE,
        'title': 'TIES Merge',
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
            {'kind': 'slider', 'name': 'Dropout', 'bind': 'dropout', 'options': {'min': 0, 'max': 1, 'step': 0.01}},
            {
                'kind': 'combo',
                'name': 'Majority Sign',
                'bind': 'majority_sign_method',
                'options': {'values': ['total', 'frequency']}
            },
        ],
        'properties': {'dropout': 0.0, 'majority_sign_method': 'total'},
        'tooltip': 'Merge PEFT tensors using the TIES algorithm (see https://arxiv.org/pdf/2306.01708).'
                   '\nDropout controls pruning before merging; majority sign selects how sign voting is performed.'
    }
