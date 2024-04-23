import qutip
import numpy as np


def _make_random_pure_state(n: int, seed: int) -> np.ndarray:
    return qutip.rand_ket(2**n, seed=seed).full().flatten()


def make_random_quantum_state(kind: str, n: int, seed: int):
    assert kind in ["pure"], kind
    if kind == "pure":
        ket = _make_random_pure_state(n, seed)
    return ket


def main():
    for n in range(1, 5 + 1):
        for seed in range(5):
            ket = make_random_quantum_state("pure", n, seed)
            print(f"{n=}, {seed=}, {ket[:5]=}, {np.linalg.norm(ket)=}")


if __name__ == "__main__":
    main()
