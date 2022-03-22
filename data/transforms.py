from typing import Dict, Hashable, Mapping, Optional
from warnings import warn

import numpy as np
from monai.config import DtypeLike, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.transform import MapTransform, Transform
from monai.transforms.utils_pytorch_numpy_unification import clip
from monai.utils import convert_data_type
from monai.utils.enums import TransformBackends


class ScaleCubedIntensityRange(Transform):
    """
    Apply specific intensity scaling to the whole numpy array.
    Scaling from [a_min, a_max] to [b_min, b_max] with clip option.

    When `b_min` or `b_max` are `None`, `scacled_array * (b_max - b_min) + b_min` will be skipped.
    If `clip=True`, when `b_min`/`b_max` is None, the clipping is not performed on the corresponding edge.

    Args:
        a_min: intensity original range min.
        a_max: intensity original range max.
        b_min: intensity target range min.
        b_max: intensity target range max.
        clip: whether to perform clip after scaling.
        dtype: output data type, if None, same as input image. defaults to float32.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        a_min: float,
        a_max: float,
        b_min: Optional[float] = None,
        b_max: Optional[float] = None,
        clip: bool = False,
        dtype: DtypeLike = np.float32,
    ) -> None:
        self.a_min = np.cbrt(a_min)
        self.a_max = np.cbrt(a_max)
        self.b_min = b_min
        self.b_max = b_max
        self.clip = clip
        self.dtype = dtype

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        img = np.cbrt(img)
        dtype = self.dtype or img.dtype
        if self.a_max - self.a_min == 0.0:
            warn("Divide by zero (a_min == a_max)", Warning)
            if self.b_min is None:
                return img - self.a_min
            return img - self.a_min + self.b_min

        img = (img - self.a_min) / (self.a_max - self.a_min)
        if (self.b_min is not None) and (self.b_max is not None):
            img = img * (self.b_max - self.b_min) + self.b_min
        if self.clip:
            img = clip(img, self.b_min, self.b_max)
        ret: NdarrayOrTensor = convert_data_type(img, dtype=dtype)[0]

        return ret


class ScaleCubedIntensityRanged(MapTransform):

    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ScaleIntensityRange`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        a_min: intensity original range min.
        a_max: intensity original range max.
        b_min: intensity target range min.
        b_max: intensity target range max.
        clip: whether to perform clip after scaling.
        dtype: output data type, if None, same as input image. defaults to float32.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = ScaleCubedIntensityRange.backend

    def __init__(
            self,
            keys: KeysCollection,
            a_min: float,
            a_max: float,
            b_min: Optional[float] = None,
            b_max: Optional[float] = None,
            clip: bool = False,
            dtype: DtypeLike = np.float32,
            allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.scaler = ScaleCubedIntensityRange(a_min, a_max, b_min, b_max, clip, dtype)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.scaler(d[key])
        return d
