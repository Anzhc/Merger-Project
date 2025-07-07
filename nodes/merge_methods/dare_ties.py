from ..utils import get_params
from utils.full_merge_utils import merge_state_dicts_with_base
from utils.peft_utils import dare_ties

NODE_TYPE = 'merge_methods/dare_ties'
NODE_CATEGORY = 'Merge method'


def execute(node, inputs):
    params = get_params(node)
    if len(inputs) < 2:
        raise ValueError('DARE TIES merge requires base and at least one model')

    dropout = float(params.get('dropout', 0.0))
    density = 1.0 - dropout
    majority = params.get('majority_sign_method', 'total')

    base = inputs[0]
    others = inputs[1:]

    base_data = base['data'] if isinstance(base, dict) else base
    dtype = base.get('dtype') if isinstance(base, dict) else None
    fmt = base.get('format', 'pt') if isinstance(base, dict) else 'pt'

    models = []
    for inp in others:
        models.append(inp['data'] if isinstance(inp, dict) else inp)

    weights = [1.0] * len(models)
    merged = merge_state_dicts_with_base(base_data, models, weights, dare_ties,
                                         density=density,
                                         majority_sign_method=majority)
    return {'data': merged, 'format': fmt, 'dtype': dtype}


def get_spec():
    return {
        'type': NODE_TYPE,
        'title': 'DARE TIES Merge',
        'category': 'merge_methods',
        'inputs': [
            {'name': 'Base', 'type': 'model'},
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
            {'kind': 'slider', 'name': 'Dropout', 'bind': 'dropout',
             'options': {'min': 0, 'max': 1, 'step': 0.01}},
            {
                'kind': 'combo',
                'name': 'Majority Sign',
                'bind': 'majority_sign_method',
                'options': {'values': ['total', 'frequency']}
            },
        ],
        'properties': {'dropout': 0.0, 'majority_sign_method': 'total'},
        'tooltip': 'Merge full models relative to Base using the DARE TIES algorithm'
    }
