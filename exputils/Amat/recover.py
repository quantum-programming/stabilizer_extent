from typing import List, Tuple

import numpy as np
from numba import njit

from exputils.Amat.get import get_Amat
from exputils.math.popcount import popcount
from exputils.math.rref import enumerate_RREF


@njit(cache=True)
def make_state_from_kQctR_sub(
    k: int, Q: int, c: int, t: int, R: np.ndarray
) -> Tuple[List[int], List[complex]]:
    indices = []
    data = []
    coeff = 1.0 / (2 ** (k / 2))
    for x in range(1 << k):
        row_idx = t
        for i in range(k):
            if x & (1 << i):
                row_idx ^= R[i]
        xQx = 0
        Q_idx = 0
        for i in range(k):
            for j in range(i, k):
                if (x & (1 << i)) and (x & (1 << j)) and (Q & (1 << Q_idx)):
                    xQx += 1
                Q_idx += 1
        indices.append(row_idx)
        cx = popcount(c & x) & 3
        if xQx & 1:
            if cx == 0:
                data.append(-coeff)
            elif cx == 1:
                data.append(-1j * coeff)
            elif cx == 2:
                data.append(+coeff)
            else:
                data.append(+1j * coeff)
        else:
            if cx == 0:
                data.append(+coeff)
            elif cx == 1:
                data.append(+1j * coeff)
            elif cx == 2:
                data.append(-coeff)
            else:
                data.append(-1j * coeff)
        # assert data[-1] == ((-1) ** (xQx)) * ((1j) ** (popcount(c & x))) * coeff
    return indices, data


def make_state_from_kQctR(
    n: int, k: int, Q: int, c: int, t: int, R: np.ndarray
) -> Tuple[List[int], List[complex]]:
    assert 0 <= Q < (1 << (k * (k + 1) // 2))
    assert 0 <= c < (1 << k) and 0 <= t < (1 << n)
    assert 0 < len(R) == k
    return make_state_from_kQctR_sub(k, Q, c, t, R)


def test_make_state_from_kQctR():
    for n in range(1, 3 + 1):
        print(n)
        Amat = get_Amat(n)
        Amat_idx = 0
        for k in range(n + 1):
            if k == 0:
                for t in range(1 << n):
                    state = np.zeros(1 << n, dtype=complex)
                    state[t] = 1.0
                    assert np.allclose(state, Amat[:, Amat_idx])
                    Amat_idx += 1
            else:
                for R, t_mask in enumerate_RREF(n, k):
                    t = 0
                    while True:
                        for Q in range(1 << (k * (k + 1) // 2)):
                            for c in range(1 << k):
                                indices, data = make_state_from_kQctR(n, k, Q, c, t, R)
                                state = np.zeros(1 << n, dtype=complex)
                                state[indices] = data
                                assert np.allclose(state, Amat[:, Amat_idx])
                                Amat_idx += 1
                        t = (t + ~t_mask + 1) & t_mask
                        if t == 0:
                            break
        print(f"test_make_state_from_kQctR({n}) passed.")


if __name__ == "__main__":
    test_make_state_from_kQctR()
