import numpy as np
from numpy import cos, sin, sqrt


def create_trajectories(N_traj=200, traj_len=1000, save_path=".", E1_rng=(0.2, 2), E2_rng=(0.2, 2), T=1e+5, noise_strength=0):
    energies1 = np.random.uniform(*E1_rng, size=N_traj)
    energies2 = np.random.uniform(*E2_rng, size=N_traj)

    A = np.sqrt(energies1)[:, None]
    B = np.sqrt(energies2)[:, None] / 2

    t = np.sort(np.random.uniform(0, T, size=(N_traj, traj_len)), axis=1)
    
    x1_arr_mode1 = x2_arr_mode1 = cos(t) * A
    p1_arr_mode1 = p2_arr_mode1 = -sin(t) * A

    x1_arr_mode2 = B * cos(sqrt(3) * t)
    x2_arr_mode2 = -B * cos(sqrt(3) * t)
    p1_arr_mode2 = -B * sqrt(3) * sin(sqrt(3) * t)
    p2_arr_mode2 = B * sqrt(3) * sin(sqrt(3) * t)

    x1_arr = x1_arr_mode1 + x1_arr_mode2
    x2_arr = x2_arr_mode1 + x2_arr_mode2
    p1_arr = p1_arr_mode1 + p1_arr_mode2
    p2_arr = p2_arr_mode1 + p2_arr_mode2

    data = np.stack((x1_arr, x2_arr, p1_arr, p2_arr), axis=2)
    noise = np.random.normal(scale=noise_strength * data.reshape(data.shape[0] * data.shape[1], -1).std(axis=0)[None, None, :], size=data.shape)
    data = data + noise
    
    if save_path is not None:
        np.savez(save_path + "/coupled_oscillator", data=data, E=np.stack((energies1, energies2), axis=1))

    return data, energies1, energies2
