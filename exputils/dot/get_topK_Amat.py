import pathlib
import numpy as np
import os
import subprocess

from exputils.dot.calc_dot import calc_dot
from exputils.dot.recovery_state import recovery_states_from_idxs
from exputils.Amat.get import get_Amat_sparse
from scipy.sparse import csc_matrix


def get_topK_Amat(n: int, psi: np.ndarray, is_dual_mode: bool) -> csc_matrix:
    assert type(psi) == np.ndarray and psi.dtype == np.complex128
    assert psi.shape == (2**n,)
    np.savez("temp_in.npz", psi=psi, is_dual_mode=is_dual_mode)

    verbose = True
    path = pathlib.Path(__file__).parent.parent / "cpp/calc_dot.exe"
    with subprocess.Popen([path], stderr=subprocess.PIPE) as p:
        if verbose:
            for line in p.stderr:
                print(line.decode("utf-8", errors="ignore"), end="")
        p.wait()
    assert p.returncode == 0, f"error in C++ code: {p.returncode=}"

    res1 = np.load("temp_out.npz")["res1"]
    res2 = np.load("temp_out.npz")["res2"]
    assert res1.dtype == np.int64 and res2.dtype == np.int64
    os.remove("temp_in.npz")
    os.remove("temp_out.npz")

    assert res2.size > 0
    if res2.size > 1 or res2[0] != -1:
        return recovery_states_from_idxs(
            n, [(int(x1) << 62) + int(x2) for x1, x2 in zip(res1, res2)]
        )
    else:
        return csc_matrix((2**n, 0))


def _get_topK_Amat_python(n: int, psi: np.ndarray, K: int) -> np.ndarray:
    assert type(psi) == np.ndarray and psi.dtype == np.complex128
    assert psi.shape == (2**n,)
    assert n <= 5

    return np.sort(np.abs(calc_dot(n, psi.tolist())))[-K:]


def _get_topK_Amat_naive(n: int, psi: np.ndarray, K: int) -> np.ndarray:
    assert type(psi) == np.ndarray and psi.dtype == np.complex128
    assert psi.shape == (2**n,)
    assert n <= 5

    Amat = get_Amat_sparse(n)
    dots = np.abs(psi.conj() @ Amat)
    idxs = np.argpartition(dots, -K)[-K:]
    return Amat[:, idxs]


def test_get_topK_Amat():
    for n in range(1, 5 + 1):
        for seed in range(3 if n <= 4 else 1):
            np.random.seed(seed)
            psi = np.random.rand(2**n) + 1j * np.random.rand(2**n)
            psi /= np.linalg.norm(psi)
            res_fast = get_topK_Amat(n, psi, False).toarray()
            res_slow = _get_topK_Amat_naive(n, psi, res_fast.shape[1])
            dots_fast = np.sort(np.abs(psi.conj() @ res_fast))
            dots_slow = np.sort(np.abs(psi.conj() @ res_slow))
            dots_python = _get_topK_Amat_python(n, psi, res_fast.shape[1])
            assert np.allclose(
                dots_slow, dots_python, atol=1e-10
            ), f"{dots_slow=}, {dots_python=}"
            assert np.allclose(
                dots_slow[-dots_fast.size :], dots_fast, atol=1e-10
            ), f"{dots_slow=}, {dots_fast=}"


if __name__ == "__main__":
    test_get_topK_Amat()
    print("get_topK_Amat.py: All tests passed")
