def succ(txt: any):
    print('\x1b[6;30;42m' + str(txt) + '\x1b[0m')


def warn(txt: str):
    print('\x1b[6;30;43m' + str(txt) + '\x1b[0m')


def err(txt: str):
    print('\x1b[6;30;41m' + str(txt) + '\x1b[0m')


def event(txt: str):
    print('\x1b[0;30;47m' + "@ " + str(txt) + '\x1b[0m')
