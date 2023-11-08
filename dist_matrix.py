import numpy as np
import wasserstein.wasserstein as ws
from sklearn.neighbors import KDTree
from tqdm import tqdm


def gen_dist_matrix(X: np.ndarray, beta=2, name=None, verbosity=1, replace_nans="auto", norm=False) -> np.ndarray:
    """
    Generates distance matrix by calculating Wasserstein distance
    @param X: Trajectories data
    @param use_sinkhorn:
    @return: Distance matrix
    """
    weighted_data = np.array([np.array([np.array([1] + list(b)) for b in a]) for a in X])
    pw_emd = ws.PairwiseEMD(beta=beta, verbose=verbosity, norm=norm)
    pw_emd(weighted_data)
    dist_matrix = pw_emd.emds()
    if replace_nans == "auto":
        val = np.nan
    else:
        val = replace_nans(dist_matrix)
    dist_matrix[dist_matrix < 0] = val
    if not (name is None):
        np.savez(name + "_dmat.npz", dmat=dist_matrix ** (1 / beta))
    return dist_matrix ** (1 / beta)
