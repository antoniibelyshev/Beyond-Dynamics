import harmonic_oscillator
import coupled_oscillator
import quantum_ho


print("generating harmonic oscillator data...")
harmonic_oscillator.create_trajectories()
print("generating coupled oscillator data...")
coupled_oscillator.create_trajectories()
print("generating quantum harmonic oscillator data...")
quantum_ho.create_trajectories()
