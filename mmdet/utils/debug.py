import torch.distributed as dist

MODE_STR = ""


def set_debug_mode(mode_str):
    global MODE_STR
    MODE_STR = mode_str.split(',')
    print('debug mode:', MODE_STR)


def is_debug(mode='DEBUG'):
    global MODE_STR
    return mode in MODE_STR


def is_master():
    if dist.get_rank() == 0:
        return True
    else:
        return False
