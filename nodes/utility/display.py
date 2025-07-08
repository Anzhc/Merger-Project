from ..utils import get_params

NODE_TYPE = 'utility/display'
NODE_CATEGORY = 'Utility'


def execute(node, inputs):
    params = get_params(node)
    value = inputs[0] if inputs else None
    if value is None:
        value = params.get('value', '')
    node.setdefault('properties', {})['value'] = str(value)
    return value


def get_spec():
    return {
        'type': NODE_TYPE,
        'title': 'Display',
        'category': 'utility',
        'inputs': [{'name': 'value', 'type': 'value'}],
        'outputs': [],
        'widgets': [
            {'kind': 'text', 'name': 'Value', 'bind': 'value', 'options': {'disabled': True}},
        ],
        'properties': {'value': ''},
        'tooltip': 'Show input value',
    }
