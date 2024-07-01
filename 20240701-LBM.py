r"""
Solves the incompressible Navier-Stokes equations using the Lattice Boltzmann Method.
The scenario is the flow around a cylinder in 2D which yields a von Karman vortex street.

                                 periodic
        +----------------------------------------------------------+
        |  ---- >>                                                 |
        |  ---- >>                                                 |
        |  ---- >>                                                 |
        |  ---- >>          ****                                   |
        |  ---- >>         *    *                                  |
 inflow |  ---- >>        *      *                                 | outflow
        |  ---- >>         *    *                                  |
        |  ---- >>          ****                                   |
        |  ---- >>                                                 |
        |  ---- >>                                                 |
        |  ---- >>                                                 |
        |  ---- >>                                                 |
        |  ---- >>                                                 |
        +----------------------------------------------------------+
                                  periodic

    -> Uniform inflow profile with only horizontal velocities at left boundary
    -> outflow boundary conditions at right boundary
    -> periodic boundary conditions at top and bottom
    -> the circle is placed at the center of the domain
    -> initially, fluid is NOT at rest and has a uniform horizontal velocity profile

Solution Strategy:
Discretize the domain into a Cartesian mesh. Each grid vortex is associated
with 9 discrete velocity components.(D2Q9) and 2 macroscopic velocity components.
The iterate over time.

1. Apply outflow boundary conditions.

2. Compute Macroscopic Quantities (density and velocity).

3. Apply Inflow profile by Zou/He profile Dirichlet boundary conditions on the left boundary.

4. Compute the discrete equilibia velocities.

5. Perform a Collision step according to BGK (Bhatnagar-Gross-Krook)

6. Apply Bounce-Back boundary conditions on the cylinder obstacle.

7. Stream-side alongsize the lattice veclocity.

8. Advance the time step.

------

Employed discretization:


D2Q9 grid, i.e. 2-dim space with 9 velocity components.

   6   2   5
    \  |  /
   3 - 0 - 1
    /  |  \
   7   4   8
Therefore we have the shapes:

- macroscoic velocity: (N_x, N_y, 2)

- discrete velocity: (N_x, N_y, 9)

- density: (N_x, N_y)

-------

Lattice-Boltzmann Computations

Density:

    -> ρ = Σᵢ fᵢ

Velocity:

    -> U = 1/ρ Σᵢ fᵢ cᵢ


Equilibrium:
fᵢᵉ = ρ Wᵢ (1+3c ⋅ U + 9c²U²/2 - 3/2||u||₂²)

BGK Collision

fᵢ₊₁ = fᵢ - ω (fᵢ - fᵢᵉ)

The relaxation factor ω is computed as:

ω = 1/(3ν + 0.5)


"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import cmasher as cmr
from tqdm import tqdm

N_ITERATIONS = 15_000
REYNOLD_NUMBER = 80

N_POINTS_X = 300
N_POINTS_Y = 50

CYLINDER_CENTER_X = N_POINTS_X // 5
CYLINDER_CENTER_Y = N_POINTS_Y // 2
CYLINDER_RADIUS_INDICES = N_POINTS_Y // 9

MAX_HORIZONTAL_INFLOW_VELOCITY = 0.04   

VISUALIZE = True
PLOT_EVERY_N_STEPS = 100
SKIP_FIRST_N_ITERATIONS = 0

r"""D2Q9 grid, i.e. 2-dim space with 9 velocity components.

   6   2   5
    \  |  /
   3 - 0 - 1
    /  |  \
   7   4   8
"""

N_DISCRETE_VELOCITIES = 9

LATTICE_VELOCITIES = jnp.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [-1, 0],
    [0, -1],
    [1, 1],
    [-1, 1],
    [-1, -1],
    [1, -1],
]).transpose();

LATTICE_INDICIES = jnp.arange(N_DISCRETE_VELOCITIES)

OPPOSITE_LATTICE_INDICIES = jnp.array([
    0, 3, 4, 1, 2, 7, 8, 5, 6
])

LATTICE_WEIGHTS = jnp.array([
    4/9,                 # center [0]
    1/9, 1/9, 1/9, 1/9,  # Axis-aligned velocities [1 2 3 4]
    1/36, 1/36, 1/36, 1/36 # 
])
LATTICE_WEIGHTS_BGK = jnp.array([
    4/9, 1/9, 1/9, 1/9, 1/9, 1/18, 1/18, 1/18, 1/18
])
