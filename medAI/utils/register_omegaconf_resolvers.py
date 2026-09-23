from datetime import datetime


_start_time = datetime.now() 


def get_start_time(format: str = "%Y-%m-%d_%H-%M-%S") -> str:
    """Resolver to get the script start time.

    Args:
        format (str): The datetime format string. Defaults to "%Y%m%d_%H%M%S".

    Returns:
        str: The formatted start time.
    """
    return _start_time.strftime(format)


def register_omegaconf_resolvers():
    try: 
        from omegaconf import OmegaConf
        OmegaConf.register_new_resolver("start_time", get_start_time)
    except ImportError:
        pass