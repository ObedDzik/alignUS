import itertools
from abc import ABC, abstractmethod
from typing import Literal, Sequence

import numpy as np
import pandas as pd
import torch


class Accumulator(ABC):
    @abstractmethod
    def __call__(self, value):
        ...

    @abstractmethod
    def reset(self):
        ...

    @abstractmethod
    def compute(self):
        ...

    def __repr__(self):
        return f"{self.__class__.__name__}({self.compute()})"


class Average(Accumulator):
    def __init__(self):
        self.reset()

    def __call__(self, value):
        self.sum += value
        self.count += 1
        return self.sum / self.count

    def reset(self):
        self.sum = 0
        self.count = 0

    def compute(self):
        return self.sum / self.count


class ExponentialMovingAverage(Accumulator):
    def __init__(self, alpha=0.9):
        self.reset()
        self.alpha = alpha

    def __call__(self, value):
        self.value = self.alpha * self.value + (1 - self.alpha) * value
        return self.value

    def reset(self):
        self.value = 0

    def compute(self):
        return self.value


class MovingAverage(Accumulator):
    def __init__(self, window_size=10):
        self.reset()
        self.window_size = window_size

    def __call__(self, value):
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)
        return sum(self.values) / len(self.values)

    def reset(self):
        self.values = []

    def compute(self):
        return sum(self.values) / len(self.values)


class Max(Accumulator):
    def __init__(self):
        self.reset()

    def __call__(self, value):
        self.value = max(self.value, value)
        return self.value

    def reset(self):
        self.value = 0

    def compute(self):
        return self.value


class Sum(Accumulator):
    def __init__(self):
        self.reset()

    def __call__(self, value):
        self.value += value
        return self.value

    def reset(self):
        self.value = 0

    def compute(self):
        return self.value


class DictConcatenation(Accumulator):
    def __init__(self):
        self.reset()

    def __call__(self, data_dict):

        for k, v in data_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu()
            elif isinstance(v, np.ndarray):
                pass
            elif not isinstance(v, Sequence):
                v = [v]
            self._data.setdefault(k, []).append(v)

    def update(self, data_dict):
        self(data_dict)

    # def compute(self, out_fmt: Literal["dict", "dataframe"] = "dict"):
    #     out = {}
    #     for k, v in self._data.items():
    #         out[k] = (
    #             torch.concat(v)
    #             if isinstance(v[0], torch.Tensor)
    #             else list(itertools.chain(*v))
    #         )

    #     for k, v in out.items():
    #         if isinstance(v, list):
    #             out[k] = np.array(v)

    #     if out_fmt == "dict":
    #         return out

    #     else:
    #         out_new = {}
    #         for k, v in out.items():
    #             if isinstance(v, torch.Tensor):
    #                 v = v.detach().cpu().numpy()
    #             if isinstance(v, np.ndarray) and v.ndim == 2:
    #                 for i in range(v.shape[1]):
    #                     out_new[f"{k}_{i}"] = v[:, i]
    #             else:
    #                 out_new[k] = v
    #         return pd.DataFrame(out_new)
        
    def compute(self, out_fmt: Literal["dict", "dataframe"] = "dataframe"):
        import numpy as np
        import pandas as pd
        import torch
        import itertools

        # print("\n[DEBUG] Starting compute()\n")

        # Step 1: Build `out`
        out = {}
        for k, v in self._data.items():
            # print(f"[DEBUG] Processing key: {k}")

            if isinstance(v[0], torch.Tensor):
                # shapes = [tuple(t.shape) for t in v]
                # print(f"  Tensor shapes before concat: {shapes}")
                out[k] = torch.concat(v)
                # print(f"  Shape after concat: {tuple(out[k].shape)}")
            else:
                # lengths = [len(x) for x in v]
                # print(f"  List lengths before flatten: {lengths}")
                out[k] = list(itertools.chain(*v))
                # print(f"  Length after flatten: {len(out[k])}")

        # Step 2: Convert lists → numpy
        for k, v in out.items():
            if isinstance(v, list):
                out[k] = np.array(v)
                # print(f"[DEBUG] Converted {k} to np.array with shape {out[k].shape}")

        # Early exit
        if out_fmt == "dict":
            # print("\n[DEBUG] Returning dict output\n")
            return out

        # Step 3: Build `out_new`
        out_new = {}
        # print("\n[DEBUG] Building out_new\n")

        for k, v in out.items():
            # print(f"[DEBUG] Processing key for DataFrame: {k}")

            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
                # print(f"  Converted tensor to numpy with shape {v.shape}")

            if isinstance(v, np.ndarray) and v.ndim == 2:
                # print(f"  Splitting 2D array with shape {v.shape}")
                for i in range(v.shape[1]):
                    col_name = f"{k}_{i}"
                    out_new[col_name] = v[:, i]
                    # print(f"    Created column {col_name} with length {len(v[:, i])}")
            else:
                # try:
                #     length = len(v)
                # except TypeError:
                #     length = "scalar"
                # print(f"  Adding column {k} with length {length}")
                out_new[k] = v

        # Step 4: Validate lengths
        # print("\n[DEBUG] Final column lengths before DataFrame:")
        lengths = {}
        for k, v in out_new.items():
            try:
                lengths[k] = len(v)
            except TypeError:
                lengths[k] = "scalar"

        # for k, l in lengths.items():
        #     print(f"  {k}: {l}")

        # Check consistency
        # numeric_lengths = [l for l in lengths.values() if isinstance(l, int)]
        # if len(set(numeric_lengths)) != 1:
        #     print("\n[ERROR] Inconsistent column lengths detected!")
        #     print(f"Unique lengths: {set(numeric_lengths)}")

        out_new.pop("topk_score", None)
        out_new.pop("entropy", None)
        # Step 5: Try creating DataFrame
        try:
            df = pd.DataFrame(out_new)
            # print("\n[DEBUG] DataFrame created successfully!\n")
            return df
        except Exception as e:
            print("\n[ERROR] DataFrame construction failed!")
            # print(str(e))
            raise

    def reset(self):
        self._data = {}


class DataFrameCollector(DictConcatenation):
    def compute(self) -> pd.DataFrame:
        return super().compute(out_fmt="dataframe")
