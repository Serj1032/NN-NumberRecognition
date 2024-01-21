import torch

_device = None

def device():
    global _device
    if _device is None:
        _device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f'Use default device: "{_device}"')
    return _device


__all__ = [
    'device'
]