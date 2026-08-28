import torch

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def _colorize_loss(_prev_loss, loss):
    if _prev_loss is None:
        color = YELLOW
    elif loss < _prev_loss:
        color = GREEN
    elif loss > _prev_loss:
        color = RED
    else:
        color = YELLOW
    return f"{color}{loss}{RESET}"

def get_device():
    if torch.cuda.is_available():
        print('[DEVICE] Discovered CUDA')
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        print('[DEVICE] Discovered MPS')
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device