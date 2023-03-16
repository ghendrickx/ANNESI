"""
Collection of decorators.

Author: Gijs G. Hendrickx
"""
import functools
import logging

"""Deprecated class/method"""

def _get_subclasses(obj):
    """Get all subclasses of an object.

    :param obj: object
    :type obj: type

    :return: all subclasses
    :rtype: list
    """
    subclasses = []
    for subclass in obj.__subclasses__():
        subclasses.append(subclass)
        subclasses.extend(_get_subclasses(subclass))
    return subclasses


def deprecated(msg):
    """Raise deprecation warning with a detailing message.

    :param msg: class-/method-specific warning message
    :type msg: str
    """

    def decorator(obj):
        """Decorator for classes/methods/functions to be deprecated.

        :param obj: to be deprecated class/object or method/function
        :type obj: type, callable

        :raises TypeError: if `obj` is not an object or function
        """
        logger = logging.getLogger(obj.__module__)

        # deprecated object
        if isinstance(obj, type):

            class DeprecatedClass(obj):
                """Deprecated class."""

                def __new__(cls, *args, **kwargs):
                    """Log a warning when a new object is generated."""
                    warn = f'Class \"{obj.__module__}.{obj.__name__}\" will be deprecated'

                    # include potential subclasses in the warning
                    subclasses = _get_subclasses(DeprecatedClass)
                    if subclasses:
                        warn += f', which includes its subclasses: '
                        for i, sub in enumerate(subclasses):
                            if i > 0:
                                warn += ', ' if i < len(subclasses) - 1 else ', and '
                            warn += f'\"{sub.__module__}.{sub.__name__}\"'

                    # append warning
                    if msg is not None:
                        warn += f'. {msg}'

                    # log warning
                    logger.warning(f'{warn}.')

                    # noinspection PyArgumentList
                    return super(DeprecatedClass, cls).__new__(cls)

            return DeprecatedClass

        # deprecated method/function
        elif callable(obj):

            @functools.wraps(obj)
            def wrapper(*args, **kwargs):
                """Wrapper."""
                # warning
                warn = f'Function \"{obj.__name__}\" will be deprecated'

                # append warning
                if msg is not None:
                    warn += f'. {msg}'

                # log warning
                logger.warning(f'{warn}.')

                # return function
                return obj(*args, **kwargs)

            return wrapper

        # invalid use of decorator
        else:
            error_msg = f'Decorator must be applied on a class/object or a method/function.'
            raise TypeError(error_msg)

    return decorator
