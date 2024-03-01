import functools
from datetime import datetime
from inspect import signature
from autotradr.config import logger


def timeit(logger=logger):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = datetime.now()
            try:
                result = func(*args, **kwargs)
            finally:
                end = (datetime.now() - start).total_seconds()
                logger.info(f"Time taken for {func.__name__}: {end:.2f} seconds")
            return result

        return wrapper

    return decorator


class ClassProperty:
    def __init__(self, method):
        self.method = method

    def __get__(self, obj, cls):
        return self.method(cls)


def classproperty(func):
    return ClassProperty(func)


class SingletonInstances(type):
    _instances = {}

    # The __call__ determines what happens when instances of SingletonInstances are called
    # Instances of SingletonInstances are classes themselves. So, when they are called, they are actually being
    # instantiated.
    # calling super().__call__ will instantiate the class if it is new/unique and return the instance
    # calling SingletonInstances() actually invokes the class-method of 'type' class and not the __call__ below
    # which has the power to create new classes
    def __call__(cls, *args, **kwargs):
        if getattr(cls, "_disable_singleton", False):
            return super().__call__(*args, **kwargs)

        sig = signature(cls.__init__)
        bound = sig.bind_partial(cls, *args, **kwargs)
        bound.apply_defaults()

        # Skip the first argument ('self') and combine args and sorted kwargs values
        sorted_kwargs_values = tuple(value for _, value in sorted(bound.kwargs.items()))
        combined_args = tuple(bound.args[1:]) + sorted_kwargs_values
        key = (cls, combined_args)

        if key not in cls._instances:
            cls._instances[key] = super().__call__(*args, **kwargs)
        return cls._instances[key]

    @classproperty
    def instances(cls):
        return cls._instances
