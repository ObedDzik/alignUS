from typing import Any, Dict
from torchvision.transforms import v2 as T


class InstanceNormalizeImage(T.Transform):
    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        mean = inpt.mean(dim=(-2, -1), keepdim=True)
        std = inpt.std(dim=(-2, -1), keepdim=True)
        return (inpt - mean) / (std + 1e-8)