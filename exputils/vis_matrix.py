import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def sort_matrix(matrix: np.ndarray) -> np.ndarray:
    assert matrix.ndim == 2 and matrix.dtype == np.complex128
    return np.array(
        sorted(
            matrix.T.tolist(),
            key=lambda x: (
                np.round(np.max(np.abs(x)), 5),
                np.round(
                    -np.abs(np.array(x).real) - np.abs(np.array(x).imag), 5
                ).tolist(),
                np.round(-np.abs(np.array(x).real), 5).tolist(),
                np.round(-np.abs(np.array(x).imag), 5).tolist(),
                np.round(-np.array(x).real, 5).tolist(),
                np.round(-np.array(x).imag, 5).tolist(),
            ),
        )
    ).T


def vis_matrix(data: np.ndarray, _title: str = "Visualization", do_sort=False) -> None:
    assert data.ndim == 2
    if do_sort:
        data = sort_matrix(data)
    vmax = max(np.abs(data).max(), 1)
    vmin = -vmax
    LRs = [(0, min(150, data.shape[1]))]
    while LRs[-1][1] < data.shape[1]:
        LRs.append((LRs[-1][1], min(LRs[-1][1] + 150, data.shape[1])))
    for i, (L, R) in enumerate(LRs):
        title = _title + "" if len(LRs) == 1 else f" ({i+1}/{len(LRs)})"
        if data.dtype == np.complex128:
            fig = plt.figure(figsize=(12, 4))
            fig.suptitle(title)
            ax1 = plt.subplot(211)
            ax1Img = ax1.imshow(data.real[:, L:R], vmin=vmin, vmax=vmax)
            divider1 = make_axes_locatable(ax1)
            fig.colorbar(ax1Img, cax=divider1.append_axes("right", size="5%", pad=0.1))
            ax1.set_title("real")
            ax1.set_xticks(range(0, R - L, 10))
            ax1.set_xticklabels(range(L, R, 10))
            ax2 = plt.subplot(212)
            ax2Img = ax2.imshow(data.imag[:, L:R], vmin=vmin, vmax=vmax)
            divider2 = make_axes_locatable(ax2)
            fig.colorbar(ax2Img, cax=divider2.append_axes("right", size="5%", pad=0.1))
            ax2.set_title("imag")
            ax2.set_xticks(range(0, R - L, 10))
            ax2.set_xticklabels(range(L, R, 10))
        else:
            fig, ax = plt.subplots(figsize=(12, 4))
            fig.suptitle(title)
            axImg = ax.imshow(data[:, L:R], vmin=vmin, vmax=vmax)
            divider = make_axes_locatable(ax)
            fig.colorbar(axImg, cax=divider.append_axes("right", size="5%", pad=0.1))
            ax.set_xticks(range(0, R - L, 10))
            ax.set_xticklabels(range(L, R, 10))
        plt.show()
