#-*-coding:utf-8-*-
import numpy as np

def interp(l, r, n_samples):
    """Intepolate between the arrays l and r, n_samples times.

    Parameters
    ----------
    l : np.ndarray
        Left edge
    r : np.ndarray
        Right edge
    n_samples : int
        Number of samples

    Returns
    -------
    arr : np.ndarray
        Inteporalted array
    """
    return np.array([
        l + step_i / (n_samples - 1) * (r - l)
        for step_i in range(n_samples)])

def make_latent_manifold(corners, n_samples):
    """Create a 2d manifold out of the provided corners: n_samples * n_samples.

    Parameters
    ----------
    corners : list of np.ndarray
        The four corners to intepolate.
    n_samples : int
        Number of samples to use in interpolation.

    Returns
    -------
    arr : np.ndarray
        Stacked array of all 2D interpolated samples
    """
    left = interp(corners[0], corners[1], n_samples)
    right = interp(corners[2], corners[3], n_samples)

    embedding = []
    for row_i in range(n_samples):
        embedding.append(interp(left[row_i], right[row_i], n_samples))
    return np.vstack(embedding)