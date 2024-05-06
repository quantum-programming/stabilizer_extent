import pathlib
import numpy as np
import os
import subprocess

from exputils.dot.get_topK_Amat import get_topK_Amat
from scipy.sparse import csc_matrix


def get_rough_topK_Amat(n: int, psi: np.ndarray) -> csc_matrix:
    assert type(psi) == np.ndarray and psi.dtype == np.complex128
    assert psi.shape == (2**n,)
    np.savez("temp_in.npz", psi=psi)

    verbose = True
    path = pathlib.Path(__file__).parent.parent / "cpp/rough_dot.exe"
    with subprocess.Popen([path], stderr=subprocess.PIPE) as p:
        if verbose:
            for line in p.stderr:
                print(line.decode("utf-8", errors="ignore"), end="")
        p.wait()
    assert p.returncode == 0, f"error in C++ code: {p.returncode=}"

    # res = np.load("temp_out.npz")["res2"]
    # os.remove("temp_in.npz")
    # os.remove("temp_out.npz")

    # assert res2.size > 0
    # if res2.size > 1 or res2[0] != -1:
    #     return recovery_states_from_idxs(
    #         n, [(int(x1) << 62) + int(x2) for x1, x2 in zip(res1, res2)]
    #     )
    # else:
    #     return csc_matrix((2**n, 0))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from exputils.state.random_ket import make_random_quantum_state

    n = 6
    psi = make_random_quantum_state("pure", n, 0)
    get_rough_topK_Amat(n, psi)

    topK_Amat = get_topK_Amat(n, psi, False)
    dots = psi.conj().T @ topK_Amat
    plt.hist(np.abs(dots), bins=100)
    plt.show()
