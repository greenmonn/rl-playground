import numpy as np


def visualize_value_function(ax,  # matplotlib axes object
                             v_pi: np.array,
                             nx: int,
                             ny: int):
    hmap = ax.imshow(v_pi.reshape(nx, ny),
                     interpolation='nearest')
    cbar = ax.figure.colorbar(hmap, ax=ax)

    # disable x,y ticks for better visibility
    _ = ax.set_xticks([])
    _ = ax.set_yticks([])

    # annotate value of value functions on heat map
    for i in range(ny):
        for j in range(nx):
            cell_v = v_pi.reshape(nx, ny)[nx - 1 - i, ny - 1 - j]
            cell_v = "{:.2f}".format(cell_v)
            ax.text(i, j, cell_v, ha="center", va="center", color="w")