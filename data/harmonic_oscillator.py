import numpy as np


def create_trajectories(N_traj=200, traj_len=1000, save_path=".", r_range=(0, 5), noise_strength=0.05):
    angles = 2 * np.pi * np.sort(np.random.uniform(size=(N_traj, traj_len)), axis=1)
    radiuses = np.random.uniform(*r_range, size=N_traj)
    
    data = radiuses[:, None, None] * np.stack((np.cos(angles), np.sin(angles)), axis=2)
    energies = radiuses ** 2 / 2

    data = data + np.random.normal(scale=noise_strength * data.std(axis=(0, 1)), size=data.shape)

    if save_path is not None:
        np.savez(save_path + "/harmonic_oscillator.npz", data=data, E=energies[:, None])

    return data, energies
