from typing import Tuple
import numpy as np

from exputils.Amat.get import get_Amat_sparse
from exputils.extent.custom import calculate_extent_custom


def calculate_extent_actual(
    n: int, psi: np.ndarray, method: str = "mosek", verbose: bool = False
) -> Tuple[float, np.ndarray, np.ndarray]:
    return calculate_extent_custom(n, get_Amat_sparse(n), psi, method, verbose)
