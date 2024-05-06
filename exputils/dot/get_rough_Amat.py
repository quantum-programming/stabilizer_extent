import pathlib
import numpy as np
import os
import subprocess

from exputils.dot.get_topK_Amat import get_topK_Amat
from scipy.sparse import csc_matrix


def get_rough_Amat(n: int, psi: np.ndarray, is_dual_mode: bool = False) -> csc_matrix:
    assert type(psi) == np.ndarray and psi.dtype == np.complex128
    assert psi.shape == (2**n,)
    np.savez("temp_in.npz", psi=psi, is_dual_mode=is_dual_mode)

    verbose = False
    path = pathlib.Path(__file__).parent.parent / "cpp/rough_dot.exe"
    with subprocess.Popen([path], stderr=subprocess.PIPE) as p:
        if verbose:
            for line in p.stderr:
                print(line.decode("utf-8", errors="ignore"), end="")
        p.wait()
    assert p.returncode == 0, f"error in C++ code: {p.returncode=}"

    loaded = np.load("temp_out.npz")
    indptr = loaded["indptr"]
    indices = loaded["indices"]
    data = loaded["data"]
    del loaded
    os.remove("temp_in.npz")
    os.remove("temp_out.npz")

    if indptr.size > 1 or indptr[0] != -1:
        return csc_matrix((data, indices, indptr), shape=(2**n, len(indptr) - 1))
    else:
        return csc_matrix((2**n, 0))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from exputils.state.random_ket import make_random_quantum_state

    n = 5
    psi = make_random_quantum_state("pure", n, 0)

    rough_Amat = get_rough_Amat(n, psi)
    dots = psi.conj().T @ rough_Amat

    topK_Amat = get_topK_Amat(n, psi, False)
    actual_dots = psi.conj().T @ topK_Amat

    plt.hist(
        [np.abs(dots), np.abs(actual_dots)],
        bins=100,
        label=["rough", "actual"],
    )
    plt.show()
