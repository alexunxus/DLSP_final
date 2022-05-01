from collections import OrderedDict

def trim_dict(state_dict, prefix = "module."):
    return OrderedDict({key.replace(prefix, ""): val for key, val in state_dict.items()})
    