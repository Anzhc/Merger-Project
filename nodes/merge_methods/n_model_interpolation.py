from ..utils import get_params

NODE_TYPE = 'merge_methods/n_model_interpolation'
NODE_CATEGORY = 'Merge method'


def execute(node, inputs):
    params = get_params(node)
    if len(inputs) < 2:
        raise ValueError('N-model interpolation requires at least two inputs')

    weights_str = params.get('weights', '')
    if weights_str:
        try:
            weights = [float(x.strip()) for x in weights_str.split(',') if x.strip()]
        except ValueError:
            raise ValueError('Weights must be a comma separated list of numbers')
    else:
        weights = []
    # default weight 1.0 for missing weights
    if len(weights) < len(inputs):
        weights.extend([1.0] * (len(inputs) - len(weights)))
    # truncate extra weights
    weights = weights[:len(inputs)]

    total = sum(weights)
    if total == 0:
        raise ValueError('Sum of weights cannot be zero')
    normalized = [w / total for w in weights]

    datas = []
    dtype = None
    fmt = 'pt'
    for inp in inputs:
        if isinstance(inp, dict):
            datas.append(inp.get('data'))
            if dtype is None:
                dtype = inp.get('dtype')
            if fmt == 'pt':
                fmt = inp.get('format', 'pt')
        else:
            datas.append(inp)

    keys = set()
    for d in datas:
        keys.update(d.keys())

    result = {}
    for k in keys:
        val = 0
        for d, w in zip(datas, normalized):
            val += d.get(k, 0) * w
        result[k] = val

    return {'data': result, 'format': fmt, 'dtype': dtype}


def get_spec():
    return {
        'type': NODE_TYPE,
        'title': 'N-Model Interpolation',
        'category': 'merge_methods',
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
            {
                'kind': 'text',
                'name': 'Weights (comma separated)',
                'bind': 'weights'
            }
        ],
        'properties': {'weights': ''},
        'tooltip': 'sum(w_i * model_i) / sum(w_i)',
    }
