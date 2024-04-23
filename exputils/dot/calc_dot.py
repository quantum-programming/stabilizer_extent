from typing import List

import numpy as np

from exputils.math.popcount import popcount
from exputils.math.rref import enumerate_RREF
from exputils.stabilizer_group import total_stabilizer_group_size


def calc_dot_sub(k: int, psi: List[complex]) -> List[complex]:
    assert 1 <= k
    assert len(psi) == (1 << k)
    if k == 1:
        return [
            psi[0] + psi[1],
            psi[0] - psi[1],
            psi[0] + 1j * psi[1],
            psi[0] - 1j * psi[1],
        ]
    else:
        ret = []

        def bfs(next: List[complex], c_0: int, q_00: int, q_1: int, i: int):
            assert 1 <= i < k
            assert c_0 == 0 or c_0 == 1
            assert q_00 == 0 or q_00 == 1
            coeff = 1j**c_0
            for q_0i in [0, 1]:
                for x1 in range(1 << (i - 1), 1 << i):
                    if q_00 ^ (popcount(q_1 & x1) & 1):
                        next[x1] = psi[(x1 << 1) + 0] - coeff * psi[(x1 << 1) + 1]
                    else:
                        next[x1] = psi[(x1 << 1) + 0] + coeff * psi[(x1 << 1) + 1]
                if i < k - 1:
                    bfs(next, c_0, q_00, q_1 ^ (q_0i << (i - 1)), i + 1)
                else:
                    ret.extend(calc_dot_sub(k - 1, next))
                coeff *= -1

        next = [0] * (1 << (k - 1))
        for c_0 in [0, 1]:
            for q_00 in [0, 1]:
                next[0] = psi[0] + ((-1) ** q_00) * (1j**c_0) * psi[1]
                bfs(next, c_0, q_00, 0, 1)

        return ret


def calc_dot(n: int, psi: List[complex]) -> List[complex]:
    assert len(psi) == (1 << n)
    assert type(psi) == list
    ret = np.zeros(total_stabilizer_group_size(n), dtype=np.complex128)
    psi = [complex(x).conjugate() for x in psi]
    ret[: len(psi)] = psi[:]  # k=0
    ret_idx = len(psi)
    for k in range(1, n + 1):
        sqrt2k = np.sqrt(2) ** k
        for mat, t_mask in enumerate_RREF(n, k):
            t = 0
            while True:
                psi2 = [0] * (1 << k)
                for x in range(1 << k):
                    row_idx = t
                    for idx in range(k):
                        if x & (1 << idx):
                            row_idx ^= mat[idx]
                    psi2[x] = psi[row_idx]
                res = calc_dot_sub(k, np.array(psi2, dtype=np.complex128))
                ret[ret_idx : ret_idx + len(res)] = res / sqrt2k
                ret_idx += len(res)
                t = (t + ~t_mask + 1) & t_mask
                if t == 0:
                    break
    assert ret_idx == len(ret), f"{idx=} {len(ret)=}"
    return ret
