from ..utils import get_params

NODE_TYPE = 'merge_methods/simple_interpolation'

def execute(node, inputs):
    params = get_params(node)
    alpha = float(params.get('alpha', 0.5))
    if len(inputs) < 2:
        raise ValueError('Interpolation node requires two inputs')
    m1, m2 = inputs[0], inputs[1]
    d1 = m1['data'] if isinstance(m1, dict) else m1
    d2 = m2['data'] if isinstance(m2, dict) else m2
    dtype = m1.get('dtype') if isinstance(m1, dict) else None
    result = {}
    for k in d1.keys():
        if k in d2:
            result[k] = (1 - alpha) * d1[k] + alpha * d2[k]
        else:
            result[k] = d1[k]
    for k in d2.keys():
        if k not in d1:
            result[k] = d2[k]
    fmt = m1.get('format', 'pt') if isinstance(m1, dict) else 'pt'
    return {'data': result, 'format': fmt, 'dtype': dtype}


def get_spec():
    """Return UI specification for this node."""
    return {
        'type': NODE_TYPE,
        'title': 'Simple Interpolation',
        'category': 'merge_methods',
        'inputs': [
            {'name': 'A', 'type': 'model'},
            {'name': 'B', 'type': 'model'},
        ],
        'outputs': [{'name': 'model', 'type': 'model'}],
        'widgets': [
            {
                'kind': 'slider',
                'name': 'Alpha',
                'bind': 'alpha',
                'options': {'min': 0, 'max': 1, 'step': 0.01},
            }
        ],
        'properties': {'alpha': 0.5},
    }
