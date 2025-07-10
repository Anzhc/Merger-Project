from ..utils import get_params

NODE_TYPE = 'utility/note'
NODE_CATEGORY = 'Utility'


def execute(node, inputs):
    params = get_params(node)
    return params.get('text', '')


def get_spec():
    return {
        'type': NODE_TYPE,
        'title': 'Note',
        'category': 'utility',
        'node_category': NODE_CATEGORY,
        'inputs': [],
        'outputs': [],
        'widgets': [
            {
                'kind': 'textarea',
                'name': 'Text',
                'bind': 'text',
            },
        ],
        'properties': {'text': ''},
        'tooltip': 'Add notes to the graph',
    }
