import pathlib
import numpy as np
import scipy.sparse
from scipy.sparse import csc_matrix


def get_Amat_sparse(n: int) -> csc_matrix:
    assert 1 <= n <= 5
    path = pathlib.Path(__file__).parent.parent.parent / f"data/Amat/Amat{n}.npz"
    return scipy.sparse.load_npz(path)


def get_Amat(n: int) -> np.ndarray:
    return get_Amat_sparse(n).toarray()


if __name__ == "__main__":
    for n in range(1, 5 + 1):
        Amat = get_Amat(n)
        print(f"Amat{n} shape: {Amat.shape}")
        if n <= 2:
            print(Amat)
