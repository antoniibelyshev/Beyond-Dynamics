from scipy.special import hermite, factorial
from scipy.integrate import trapezoid
import numpy as np
from typing import Callable, Tuple
from tqdm import trange


def momentum_wfunc(x: np.ndarray, psi: np.ndarray,  position_shift: int = None, momentum_shift: int = None) -> Tuple[np.ndarray, np.ndarray]:
    if position_shift is None:
        position_shift = len(x) // 2
    if momentum_shift is None:
        momentum_shift = len(x) // 2
    dx = x[1] - x[0]
    dp = 2 * np.pi / dx / len(x)
    p = (np.arange(len(x)) - momentum_shift) * dp

    psi_momentum = np.roll(np.fft.fft(np.roll(psi, position_shift)), momentum_shift)
    psi_momentum = psi_momentum / trapezoid(psi_momentum, p)

    return p, psi_momentum


def compute_qho_bfuncs(n_basis: int, x: np.ndarray) -> np.ndarray:
    res = np.zeros((n_basis, len(x)))
    for i in range(n_basis):
        factor = 1 / np.sqrt((2 ** i) * factorial(i) * np.sqrt(np.pi))
        res[i] = hermite(i)(x) * np.exp(-x ** 2 / 2) * factor
    return res


def energies(n_basis: int) -> np.ndarray:
    return np.arange(n_basis) + 0.5


def compute_wfunc(
        init_wfunc: np.ndarray,
        n_basis: int,
        x: np.ndarray,
        t: np.ndarray,
        eval_bfuncs: Callable[[int, np.ndarray], np.ndarray] = compute_qho_bfuncs,
    ) -> np.ndarray:
    init_wfunc = init_wfunc[None, :, None]

    bfuncs = eval_bfuncs(n_basis, x)[..., None]
    
    d = x[1:] - x[:-1]
    int_coefs = (np.concatenate((d, [0])) + np.concatenate(([0], d))) / 2
    int_coefs = int_coefs[None, :, None]
    coeffs = (init_wfunc * bfuncs * int_coefs).sum(axis=1, keepdims=True)

    E = energies(n_basis)[:, None, None]
    time_exps = np.exp(-1j * E * t[None, None, :])
    
    wfunc = (coeffs * bfuncs * time_exps).sum(axis=0).T
    
    return wfunc


class QHO:
    def __init__(self, n_basis: int = 40, grid_size: int = 1000):
        self.n_basis = n_basis
        self.grid_size = grid_size
        self.dx = self.dp = np.sqrt(2 * np.pi / grid_size)
        self.x = self.get_grid(self.dx)
        self.p = self.get_grid(self.dp)
        self.bfuncs = self.get_bfuncs()

    def get_grid(self, d) -> np.ndarray:
        L = d * self.grid_size
        return np.linspace(-L / 2, L / 2, self.grid_size)

    def get_bfuncs(self) -> np.ndarray:
        res = np.zeros((self.n_basis, self.grid_size))
        for i in range(self.n_basis):
            factor = 1 / np.sqrt((2 ** i) * factorial(i) * np.sqrt(np.pi))
            res[i] = hermite(i)(self.x) * np.exp(-self.x ** 2 / 2) * factor
        return res
    
    def get_coeffs(self, init_wfun: np.ndarray) -> np.ndarray:
        coeffs = self.dx * (init_wfun * self.bfuncs).sum(axis=1)
        return coeffs
    
    def get_E(self) -> np.ndarray:
        return np.arange(self.n_basis) + 0.5
    

    def get_position_wfun(self, init_wfun: np.ndarray, t: float) -> np.ndarray:
        coeffs = self.get_coeffs(init_wfun)[:, None]
        E = self.get_E()[:, None]
        series = coeffs * self.bfuncs * np.exp(-1j * E * t)
        wfun = series.sum(axis=0)
        return wfun

    def get_momentum_wfun(self, position_wfun: np.ndarray) -> np.ndarray:
        shifted_position_wfun = np.roll(position_wfun, self.grid_size // 2)
        shifted_momentum_wfun = np.fft.fft(shifted_position_wfun)
        momentum_wfun = np.roll(shifted_momentum_wfun, self.grid_size // 2)
        momentum_wfun /= (np.abs(momentum_wfun) ** 2).sum() * self.dp
        return momentum_wfun
    
    def sample_point(self, init_wfun, t):
        position_wfun = self.get_position_wfun(init_wfun, t)
        momentum_wfun = self.get_momentum_wfun(position_wfun)
        x = self.sample(position_wfun, self.x)
        p = self.sample(momentum_wfun, self.p)
        return np.array([x, p])
    
    def sample(self, wfun: np.ndarray, vals: np.ndarray):
        p = np.abs(wfun) ** 2
        p = p / p.sum()
        return np.random.choice(vals, p=p)
    

class VariatingInitialCondition:
    def __init__(self, x: np.ndarray, x0: float, pos_var: float = 0,
                 var_var: float = 0):
        self.x = x
        self.x0 = x0
        self.pos_var = pos_var
        self.var_var = var_var

    def get_init_wfun(self) -> np.ndarray:
        x0 = np.random.normal(self.x0, self.pos_var)
        sigma = np.random.normal(1, self.var_var)
        coef = (np.pi * sigma) ** (-0.25)
        return np.exp(-(self.x - x0) ** 2 / (2 * sigma)) * coef


def create_trajectories(N_traj=200, traj_len=1000, save_path=".", x0_range=(0, 5), init_variations=(0.1, 0.1)):
    qho = QHO()

    t_max = 2 * np.pi
    t = np.linspace(0, t_max, traj_len)

    data = np.zeros((N_traj, traj_len, 2))
    x0 = np.random.uniform(*x0_range, size=N_traj)
    for i in trange(N_traj):
        vic = VariatingInitialCondition(qho.x, x0[i], *init_variations)
        for j in range(traj_len):
            init_wfun = vic.get_init_wfun()
            data[i][j] = qho.sample_point(init_wfun, t[j])

    if save_path is not None:
        np.savez(save_path + "/quantum_ho.npz", data=data, x0=x0[:, None])

    return data

    # qho = QHO()

# N = 200

# N_t = 1000
# t_max = 2 * np.pi
# t = np.linspace(0, t_max, N_t)

# pos_var = 0.1
# var_var = 0.1

# X = np.zeros((N, N_t, 2))
# x0 = np.random.uniform(0, 5, size=N)
# for i in trange(N):
#     vic = VariatingInitialCondition(qho.x, x0[i], pos_var, var_var)
#     for j in range(N_t):
#         init_wfun = vic.get_init_wfun()
#         X[i][j] = qho.sample_point(init_wfun, t[j])


# X = StandardScaler().fit_transform(np.concatenate(list(X))).reshape(X.shape)
# dmat = utils.gen_dist_matrix(X)
