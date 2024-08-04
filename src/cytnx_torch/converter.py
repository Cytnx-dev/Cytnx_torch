from typing import List, TYPE_CHECKING
import numpy as np
from dataclasses import dataclass
from abc import abstractmethod
from .bond import Bond, SymBond

if TYPE_CHECKING:
    from .unitensor.entry import RegularUniTensor, BlockUniTensor
    from .unitensor.base import AbstractUniTensor


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
        assert len(set(self.input_labels).union(self.output_labels)) == len(
            self.input_labels
        ) + len(self.output_labels), "error, contain duplicate labels."

    def revert(self) -> "AbstractConverter":
        return self.__class__(
            input_bonds=[bond.redirect() for bond in self.output_bonds],
            input_labels=self.output_labels,
            output_bonds=[bond.redirect() for bond in self.input_bonds],
            output_labels=self.input_labels,
        )

    @abstractmethod
    def _contract(
        self, is_lhs: bool, utensor: "AbstractUniTensor"
    ) -> "AbstractUniTensor":
        raise NotImplementedError("not implement for abstract type trait.")

    def contract(self, utensor: "AbstractUniTensor") -> "AbstractUniTensor":

        # label:
        if not set(self.input_labels).issubset(set(utensor.labels)):
            raise ValueError("The input labels are not compatible with the tensor.")

        # check bond:
        for i, lbl in enumerate(self.input_labels):
            if not self.input_bonds[i].contractable_with(utensor.get_bond(lbl)):
                raise ValueError(
                    "The bond with label [label [{lbl}] @ idx=[{i}] with are not compatible with the bond in tensor."
                )

        return self._contract(is_lhs=True, utensor=utensor)


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

    def _contract(
        self, is_lhs: bool, utensor: "RegularUniTensor"
    ) -> "RegularUniTensor":

        if utensor.is_diag:
            raise ValueError("error, cannot contract with diagonal tensor.")

        remaind_labels = [x for x in utensor.labels if x not in self.input_labels]

        p_labels = (
            self.input_labels + remaind_labels
            if is_lhs
            else remaind_labels + self.input_labels
        )

        permuted_ut = utensor.permute(*p_labels, by_label=True)

        if is_lhs:
            new_labels = self.output_labels + remaind_labels
            new_shape = [bond.dim for bond in self.output_bonds] + list(
                permuted_ut.shape[len(self.input_labels) :]
            )
            new_bonds = self.output_bonds + permuted_ut.bonds[len(self.input_labels) :]
        else:
            new_labels = remaind_labels + self.output_labels
            new_shape = list(permuted_ut.shape[: len(remaind_labels)]) + [
                bond.dim for bond in self.output_bonds
            ]
            new_bonds = permuted_ut.bonds[: len(remaind_labels)] + self.output_bonds

        permuted_ut._get_generic_meta()

        new_data = permuted_ut.data.reshape(new_shape)

        return utensor.__class__(
            labels=new_labels,
            bonds=new_bonds,
            backend_args=permuted_ut.backend_args,
            name=permuted_ut.name,
            rowrank=len(new_labels) // 2,
            data=new_data,
        )


@dataclass
class BlockUniTensorConverter(AbstractConverter):

    def __post_init__(self):
        super().__post_init__()
        assert np.all(
            [isinstance(bond, SymBond) for bond in self.input_bonds]
        ), "error, not all input bonds are Symmetry Bond."
        assert np.all(
            [isinstance(bond, SymBond) for bond in self.output_bonds]
        ), "error, not all output bonds are Symmetry Bond."

    def _contract(self, is_lhs: bool, utensor: "BlockUniTensor") -> "BlockUniTensor":

        raise NotImplementedError("TODO")
