import qutip
import numpy as np


def _make_random_pure_state(n: int, seed: int) -> np.ndarray:
    return qutip.rand_ket(2**n, seed=seed).full().flatten()


def make_random_quantum_state(kind: str, n: int, seed: int) -> np.ndarray:
    assert kind in ["pure", "real"], kind
    if kind == "pure":
        ket = _make_random_pure_state(n, seed)
    elif kind == "real":
        np.random.seed(seed)
        ket = np.random.rand(2**n) - 0.5
        ket = ket / np.linalg.norm(ket)
        ket = ket.astype(np.complex128)

    return ket


def main():
    for n in range(1, 5 + 1):
        for seed in range(5):
            for kind in ["pure", "real"]:
                ket = make_random_quantum_state(kind, n, seed)
                print(f"{n=}, {seed=}, {kind=}, {ket[:5]=}, {np.linalg.norm(ket)=}")


if __name__ == "__main__":
    main()
