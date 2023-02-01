import unittest
import time
from threading import Timer


def debounce(wait):
    """ Decorator that will postpone a functions
        execution until after wait seconds
        have elapsed since the last time it was invoked. """
    def decorator(fn):
        def debounced(*args, **kwargs):
            def call_it():
                fn(*args, **kwargs)
            try:
                debounced.t.cancel()
            except (AttributeError):
                pass
            debounced.t = Timer(wait, call_it)
            debounced.t.start()
        return debounced
    return decorator


count = 0


@debounce(0.5)
def increment():
    global count
    """ Simple function that
        increments a counter when
        called, used to test the
        debounce function decorator """
    count += 1
    print(count)


if __name__ == '__main__':
    while (True):
        a = input()
        increment()
