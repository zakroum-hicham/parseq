from .system import PARSeq


class InvalidModelError(RuntimeError):
    """Exception raised for any model-related error (creation, loading)"""



def load_from_checkpoint(checkpoint_path: str, **kwargs):
    parseq = PARSeq()
    model = parseq.load_from_checkpoint(checkpoint_path, **kwargs)
    return model


def parse_model_args(args):
    kwargs = {}
    arg_types = {t.__name__: t for t in [int, float, str]}
    arg_types['bool'] = lambda v: v.lower() == 'true'  # special handling for bool
    for arg in args:
        name, value = arg.split('=', maxsplit=1)
        name, arg_type = name.split(':', maxsplit=1)
        kwargs[name] = arg_types[arg_type](value)
    return kwargs