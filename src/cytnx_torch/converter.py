import numpy as np
from dataclasses import dataclass, field
from .unitensor import AbstractUniTensor


@dataclass
class Converter(AbstractUniTensor):

    _look_ups: np.ndarray = field(
        default_factory=lambda: np.ndarray(shape=(0, 0), dtype=np.int64)
    )

    def __post_init__(self):
        if self._look_ups.shape[1] != len(self.bonds):
            raise ValueError(
                f"number of bonds should be equal to number of look_ups. got {self._look_ups.shape[1]} and {len(self.bonds)}"
            )
