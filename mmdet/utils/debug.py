MODE_STR = ""


def set_debug_mode(mode_str):
    global MODE_STR
    MODE_STR = mode_str.split(',')
    print('debug mode:', MODE_STR)


def is_debug(mode='DEBUG'):
    global MODE_STR
    return mode in MODE_STR
