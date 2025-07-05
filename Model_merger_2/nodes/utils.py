
def get_params(node):
    """Return parameters stored on a node"""
    return node.get('params') or node.get('properties', {})
