import torch


__all__ = ["SelectDictKeys"]


class SelectDictKeys: 
    def __init__(self, keys):
        self.keys = keys
    
    def __call__(self, sample): 
        output = []
        for key in self.keys:
            if key in sample:
                output.append(sample[key])
        if len(output) == 1:
            return output[0]
        else: 
            return tuple(output)


class AddDummyTarget: 
    def __init__(self, target=0, key='label'):
        self.target = target
        self.key = key

    def __call__(self, sample): 
        target = torch.tensor(self.target)
        if isinstance(sample, dict):
            sample[self.key] = target
            return sample
        else: 
            return (sample, target)


class AddAliasKeyToDict: 
    def __init__(self, original_key, alias_key):
        self.original_key = original_key
        self.alias_key = alias_key

    def __call__(self, sample): 
        if isinstance(sample, dict) and self.original_key in sample:
            sample[self.alias_key] = sample[self.original_key]
        return sample