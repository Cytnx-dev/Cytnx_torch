from typing import List, Optional, Callable, Type
import numpy as np
from dataclasses import dataclass, field
from .unitensor import RegularUniTensor, BlockUniTensor
from .bond import Bond, BondType


@dataclass
class AbstractConverter:

    input_bonds: List[Bond]
    input_labels: List[str]
    output_bonds: List[Bond]
    output_labels: List[str]

    def __post_init__(self):
        assert len(self.input_bonds) == len(
            self.input_labels
        ), "error, the number of input bonds and input labels are not the same."
        assert len(self.output_bonds) == len(
            self.output_labels
        ), "error, the number of output bonds and output labels are not the same."
        assert len(set(self.input_labels) + set(self.output_labels)) == len(
            self.input_labels
        ) + len(self.output_labels), "error, contain duplicate labels."


@dataclass
class RegularUniTensorConverter(AbstractConverter):

    def __post_init__(self):
        super().__post_init__()
        assert np.all(
            [isinstance(bond, Bond) for bond in self.input_bonds]
        ), "error, not all input bonds are regular Bond."
        assert np.all(
            [isinstance(bond, Bond) for bond in self.output_bonds]
        ), "error, not all output bonds are regular Bond."

    def _lcontract(self, utensor: RegularUniTensor) -> RegularUniTensor:

        p_labels = self.output_labels + [
            x for x in utensor.labels if not x in self.input_labels
        ]

        permuted_ut = utensor.permute(*p_labels, by_label=True)

        pass

    def _rcontract(self, utensor: RegularUniTensor) -> RegularUniTensor:
        pass

    def contract(self, utensor: RegularUniTensor) -> "RegularUniTensor":
        # check if the input bonds are compatible with the tensor

        # label:
        if not set(self.input_labels).issubset(set(utensor.labels)):
            raise ValueError("The input labels are not compatible with the tensor.")

        return self._lcontract(utensor)


@dataclass
class BlockUniTensorCombiner(AbstractConverter):
    pass
