
from typing import Any, Dict, List
from torchvision.transforms import v2 as T
from medAI.utils.data import get_crop_to_mask_params


class CropToMask(T.Transform): 

    def __init__(self, reference_mask_key, padding=0): 
        super().__init__()
        self.reference_mask_key = reference_mask_key
        self.padding = padding

    def forward(self, *inputs: Any) -> Any:
        if not isinstance(inputs[0], dict): 
            raise ValueError("Expected dict for this transform")
        inputs[0][self.reference_mask_key]._is_reference_mask = True
        return super().forward(*inputs)

    def _is_reference_mask(self, inp): 
        return getattr(inp, "_is_reference_mask", False)

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        reference_mask = [inp for inp in flat_inputs if self._is_reference_mask(inp)][0]
        try: 
            top, left, height, width = get_crop_to_mask_params(reference_mask[0].shape, reference_mask[0], padding=self.padding)
        except: 
            # it is possible for there to be no mask. In that case, we should be a no-op.
            return {}

        return dict(
            top=top, 
            left=left, 
            height=height, 
            width=width
        )

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if not params: 
            return inpt
        return T.functional.crop(
            inpt, **params
        )