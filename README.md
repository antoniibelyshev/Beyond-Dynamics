# Beyond-Dynamics

## Abstract

The discovery of conservation principles is crucial for understanding the underlying behavior of physical systems and has applications across various domains. In this paper, we propose a novel method that combines representation learning and topological analysis to uncover the topology of conservation law spaces. Our approach is robust, as it does not rely on expert-selected values of hyperparameters, making it accessible to researchers from different disciplines. We demonstrate the method's effectiveness on a set of physical simulations, showcasing its potential for uncovering previously unknown conservation principles and fostering interdisciplinary research. Ultimately, this work emphasizes the power of data-driven techniques in advancing our understanding of fundamental principles governing physical systems and beyond.

## Methods

### Data

In our work we learn to determine the number of conserved quantities in dynamical system using the data from trajectories of this system. For a given model we consider a dataset of $N$ trajectories, where each trajectory is represented by $M$ points in the phase space. Initial conditions for the trajectories are drawn randomly from the phase space. Our algorithm only works under the assumption that the time of sampling is sufficiently large, such that any two trajectories with equal conserved quantities will draw from a single distribution which depend only on the values of the conserved quantities and not on the specific initial condition. The property of the system to have such sufficiently large time is called ergodicyty, so we consider only ergodic systems.

### Algorithm

We start the algorithm with normalizing the data to have the zero mean and unit variance along each individual coordinate of the phase space. To the resulting rescaled trajectories we apply the Wasserstein distance to compute pairwise distances between trajectories. These pairwise distances approximate the metric sctructure of the manifold $\mathcal{C}$ consisting of all possible trajectories of the system. Then we construct a series of manifold approximations for in various dimensionalities. The minimal dimension where the approximations *is good*, corresponds to the dimensionality of the manifold $\mathcal{C}$. Since, $\mathcal{C}$ can be parametrised by the values of conserved quantities, so its dimensionality equals to the number of conserved quantities. Therefore, our algorithm is able to find the number of conserved quantities in the system.

## How to run the code

All nessecary packages can be installed by running "pip install -r requirements.txt" in the terminal.
For our experiments we use simulated data. The scripts for the data generation are in the ./data.
The experiments are shown in the respective notebooks.
