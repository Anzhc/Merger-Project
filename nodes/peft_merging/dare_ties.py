from ..utils import get_params
from utils.peft_utils import merge_state_dicts, dare_ties
# Inspired by HuggingFace PEFT implementations

NODE_TYPE = 'peft_merging/dare_ties'
NODE_CATEGORY = 'PEFT merging'


def execute(node, inputs):
    params = get_params(node)
    if len(inputs) < 2:
        raise ValueError('DARE TIES merge requires at least two inputs')

    dropout = float(params.get('dropout', 0.0))
    density = 1.0 - dropout
    majority = params.get('majority_sign_method', 'total')
    rescale = bool(params.get('rescale', True))

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
    merged = merge_state_dicts(
        models,
        weights,
        dare_ties,
        density=density,
        majority_sign_method=majority,
        rescale=rescale,
    )
    return {'data': merged, 'format': fmt, 'dtype': dtype}


def get_spec():
    return {
        'type': NODE_TYPE,
        'title': 'DARE TIES Merge',
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
            {
                'kind': 'checkbox',
                'name': 'Rescale After Pruning',
                'bind': 'rescale'
            },
        ],
        'properties': {
            'dropout': 0.0,
            'majority_sign_method': 'total',
            'rescale': True,
        },
        'tooltip': 'Combine PEFT tensors with DARE TIES algorithm (https://arxiv.org/pdf/2311.03099v3).\nDropout is applied before sign-based merging. If rescale is enabled, weights are divided by density after pruning.'
    }
