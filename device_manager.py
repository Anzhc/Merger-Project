import torch

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
    return _default_device


def set_device(device: str):
    global _default_device
    if device not in available_devices():
        raise ValueError(f'Unsupported device: {device}')
    _default_device = device
