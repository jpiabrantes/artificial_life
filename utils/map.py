import noise
import numpy as np


def create_tiles_1(n_rows, n_cols, max_capacity):
    """
    Create food capacity.

    :param n_rows:
    :param n_cols:
    :return:
    """
    shape = (n_rows, n_cols)
    scale = 5.0
    octaves = 6
    persistence = 0.5
    lacunarity = 2

    pnoise = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            pnoise[i][j] = noise.pnoise2(i / scale, j / scale, octaves=octaves, persistence=persistence,
                                         lacunarity=lacunarity,
                                         repeatx=50,
                                         repeaty=50,
                                         base=0)

    capacity = np.zeros_like(pnoise)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if -0.1 < pnoise[i][j] < 0:
                capacity[i][j] = max_capacity

    return capacity


if __name__ == '__main__':
    import scipy.misc
    max_capacity = 2
    growth_rate = 0.1
    dirt = np.array((120., 72, 0)) / 255
    grass = np.array((85., 168, 74)) / 255

    sugar_norm = create_tiles_1(50, 50, max_capacity)[..., None] / max_capacity

    rgb = grass * sugar_norm + (1 - sugar_norm) * dirt

    print('Max capacity:', np.sum(sugar_norm != 0)*max_capacity)
    print('Sustained capacity:', np.sum(sugar_norm != 0) * growth_rate)
    scipy.misc.imshow(scipy.misc.imresize(rgb, (1024, 1024), interp='nearest'))



