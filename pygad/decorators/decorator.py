def static_vars(**kwargs):
    '''
    Decorate a function with static variables.

    Example:
        >>> @static_vars(counter=0)
        ... def foo():
        ...     foo.counter += 1
        ...     print('foo got called the %d. time' % foo.counter)
        >>> foo()
        foo got called the 1. time
        >>> foo()
        foo got called the 2. time
        >>> foo()
        foo got called the 3. time
    '''

    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate

