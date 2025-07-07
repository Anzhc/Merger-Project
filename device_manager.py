import torch

# Track devices that have been used by the app. Start with CPU
_used_devices = {'cpu'}

_default_device = 'cpu'


def available_devices():
    devices = ['cpu']
    try:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(f'cuda:{i}')
    except Exception:
        pass
    return devices


def get_device():
    _used_devices.add(_default_device)
    return _default_device


def set_device(device: str):
    global _default_device
    if device not in available_devices():
        raise ValueError(f'Unsupported device: {device}')
    _default_device = device
    _used_devices.add(device)


def mark_device_used(device: str):
    """Register a device as used for the current run."""
    _used_devices.add(device)


def cleanup_memory():
    """Release cached memory on devices that were used by this app."""
    try:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            current = torch.cuda.current_device()
            for dev in list(_used_devices):
                if dev.startswith('cuda'):
                    try:
                        torch.cuda.set_device(dev)
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                    except Exception:
                        pass
            torch.cuda.set_device(current)
    except Exception:
        pass
    finally:
        _used_devices.clear()
        _used_devices.add('cpu')
