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
        