from torchvision.transforms import v2 as T


class ApplyTransformToKeys:
    def __init__(self, transform, keys, output_keys=None):
        self.transform = transform
        self.keys = keys
        self.output_keys = output_keys or keys

    def __call__(self, sample):
        for key, output_key in zip(self.keys, self.output_keys):
            sample[output_key] = self.transform(sample[key])
        return sample


class DuplicateKeys:
    def __init__(self, keys, copies=2):
        self.keys = keys
        self.copies = copies

    def __call__(self, sample):
        for key in self.keys:
            sample[key] = [sample[key] for _ in range(self.copies)]
        return sample


class ApplyNTimes:
    def __init__(self, transform, n):
        self.transform = transform
        self.n = n

    def __call__(self, sample):
        return [self.transform(sample) for _ in range(self.n)]