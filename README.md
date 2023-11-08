# ICLR Supplementary materials

## Data sampling details

In the oscillating Turing patterns we generated 200 trajectories with 400 points in each trajectory. For other experiments we generated 200 trajectories with 1000 points in each trajectory.

To simulate a noise in measurments, we modified the noiseless harmonic oscillator measurments by adding to each point a random normal noise with zero mean and variance equal to the variance along the corresponding coordinate multiplied by the strength of the noise, in our case equal to 0.01.

For the quantum harmonic poscillator we considered the following experimental setup for getting the data for our algorithm. Experiment starts with the wavefunction being a gaussian with the mean equal to some $x_0$ and variance equal to $1 / \sqrt{2}$, then the wavefunction evolves for some time $t$, and at last we make a measurment of either position or momentum, which is just getting a sample from a probability distribution $\rho(y) = |\psi(y, t)|^2$, where $y$ is either $x$ or $p$ depending on what we are currently measuring. After one measurment is done we have to restart the experiment because the measurment of the quantum system can change its state. In order to measure the whole trajectory, we have to repeat the experiment with the same initial conditions as many times as many measurments we want to be made. The process repeats with the different $x_0$ sampled randomly in the interval [0, 5]. To make this setup more realistic, we add some error in the initial condition for each new experiment: both mean and variance of the gaussian are normally distributed with means $x_0$ and 1 respectively and with variances equal to 0.1.

## Running the code

To generate the trajectories you can run the script generate_trajectories.py in the folder data.

After that you can run the experiments from the paper using the corresponding notebooks in the supplementary materials.