# ICLR Supplementary materials

## Data sampling details

In the oscillating Turing patterns we generated 200 trajectories with 400 points in each trajectory. For other experiments we generated 200 trajectories with 1000 points in each trajectory.

To simulate a noise in measurments, we modified the noiseless harmonic oscillator measurments by adding to each point a random normal noise with zero mean and variance equal to the variance along the corresponding coordinate multiplied by the strength of the noise, in our case equal to 0.01.

For the quantum harmonic poscillator we considered the following experimental setup for getting the data for our algorithm. Experiment starts with the wavefunction being a gaussian with the mean equal to some $x_0$ and variance equal to $1 / \sqrt{2}$, then the wavefunction evolves for some time $t$, and at last we make a measurment of either position or momentum, which is just getting a sample from a probability distribution $\rho(y) = |\psi(y, t)|^2$, where $y$ is either $x$ or $p$ depending on what we are currently measuring. After one measurment is done we have to restart the experiment because the measurment of the quantum system can change its state. In order to measure the whole trajectory, we have to repeat the experiment with the same initial conditions as many times as many measurments we want to be made. The process repeats with the different $x_0$ sampled randomly in the interval [0, 5]. To make this setup more realistic, we add some error in the initial condition for each new experiment: both mean and variance of the gaussian are normally distributed with means $x_0$ and 1 respectively and with variances equal to 0.1.

## Running the code

To generate the trajectories you can run the script generate_data.py in the folder data.

After that you can run the experiments from the paper using the corresponding notebooks in the supplementary materials.

## Additional experiments

In addition to the experiments presented in the paper we performed two additional experiments, to show how our algorithm works with various levels of noise and how it works with the data where one of the conserved quantities is almost the same for all trajectories in the input data.

### Coupled oscillator with various levels of noise

We tested our algorithm on the coupled oscillator introducing various levels of noise into the normalized data (that corresponds to the unnirmalized data with noise scaled by the standard deviation of the particular coordinate in the data). You can find the results of the experiment in the file "msap_coupled_oscillator_noise.ipynb". On the resulting graph we see that our algorithm distinguishes the 2-dimensional embedding from the 1-dimensional embedding up to the noise std equal to 0.5, however, the distinction becomes less obvious as the noise grows.

### Coupled oscillator with small variation in $E_2$

In this experiment we sample the initial points for the trajectories such that the $E_1$ varies from 0.2 to 2.0 and $E_2$ varies from 1.0 to 1.1 (variation in $E_2$ is much smaller than variation in $E_1$). We run our algorithm on such data. The results of the experiment are shown in the notebook "msap_coupled_oscillator_fixed_E2.ipynb". Our algorithm showed that there is one conserved quantity in the system instead of the actual two conserved quantities, because the input data did not represent the space of conserved quantities, the shape space $\mathcal{C}$, well. Also, latent representations have learnt the $E_1$ and have not learnt $E_2$, because there is almost no usefull information about the structure of $E_2$.