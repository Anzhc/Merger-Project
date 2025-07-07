
import device_manager


def get_params(node):
    """Return parameters stored on a node."""
    return node.get('params') or node.get('properties', {})


def get_device(node=None):
    """Return device for a node or global default."""
    if node:
        params = get_params(node)
        dev = params.get('device')
        if dev:
            device_manager.mark_device_used(dev)
            return dev
    return device_manager.get_device()
