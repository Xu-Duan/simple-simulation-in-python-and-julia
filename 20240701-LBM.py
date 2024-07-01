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

- macroscopic velocity: (N_x, N_y, 2)

- discrete velocity: (N_x, N_y, 9)

- density: (N_x, N_y)

-------

Lattice-Boltzmann Computations

Density:

    -> ρ = Σᵢ fᵢ

Velocity:

    -> U = 1/ρ Σᵢ fᵢ cᵢ


Equilibrium:
fᵢᵉ = ρ Wᵢ (1+3c ⋅ U + 9(c ⋅ U)²/2 - 3/2||u||₂²)

BGK Collision

fᵢ₊₁ = fᵢ - ω (fᵢ - fᵢᵉ)

The relaxation factor ω is computed as:

ω = 1/(3ν + 0.5)

where:
    fᵢ: discrete velocities
    U: macroscopic velocities
    cᵢ: lattice velocities

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
SKIP_FIRST_N_ITERATIONS = 5000

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

RIGHT_VELOCITIES = jnp.array([1, 5, 8])
UP_VELOCITIES = jnp.array([2, 5, 6])
LEFT_VELOCITIES = jnp.array([3, 6, 7])
DOWN_VELOCITIES = jnp.array([4, 7, 8])
PURE_VERTICAL_VELOCITIES = jnp.array([0, 2, 4])
PURE_HORIZONTAL_VELOCITIES = jnp.array([0, 1, 3])

def get_density(discrete_velocities):
    return jnp.sum(discrete_velocities, axis=-1)

def get_macroscopic_velocities(discrete_velocities, density):
    macroscopic_velocities = jnp.einsum(
            "NMQ,dQ->NMd", discrete_velocities, LATTICE_VELOCITIES
            ) / density[..., jnp.newaxis]
    
    return macroscopic_velocities

def get_equilibrium_discrete_velocities(macroscopic_velocities, density):
    projected_discrete_velocities = jnp.einsum(
            "dQ, NMd->NMQ", LATTICE_VELOCITIES, macroscopic_velocities
            )
    macroscopic_velocity_magnitude = jnp.linalg.norm(macroscopic_velocities, axis=-1,
                                                     ord=2)
    equilibrium_discrete_velocities = (
            density[..., jnp.newaxis]
            *
            LATTICE_WEIGHTS[jnp.newaxis, jnp.newaxis, :]
            *
            (
                1+
                3 * projected_discrete_velocities
                +
                9/2 * projected_discrete_velocities**2
                -
                3/2 * macroscopic_velocity_magnitude[..., jnp.newaxis]**2
            )
            )
    
    return equilibrium_discrete_velocities

def main():
    jax.config.update("jax_enable_x64", True)
    # Re = UR/ν => ν = UR/Re
    kinematic_viscosity = (
            MAX_HORIZONTAL_INFLOW_VELOCITY
            *
            CYLINDER_RADIUS_INDICES
            /
            REYNOLD_NUMBER
        )
    relaxation_omega = (
        1.0 / (
            3.0 * kinematic_viscosity
            + 0.5
        )
    )

    # define mesh
    x = jnp.arange(N_POINTS_X)
    y = jnp.arange(N_POINTS_Y)
    X, Y = jnp.meshgrid(x, y, indexing = "ij")

    # obstable Mask: An array of the shape like X or Y, but contains True if the point
    # belong to the obstacle and False otherwise.
    obstacle_mask = (
        (X - CYLINDER_CENTER_X) ** 2
        + (Y - CYLINDER_CENTER_Y) ** 2
        <= CYLINDER_RADIUS_INDICES
    )
    
    velocity_profile = jnp.zeros((N_POINTS_X, N_POINTS_Y, 2))
    velocity_profile = velocity_profile.at[:, :, 0].set(MAX_HORIZONTAL_INFLOW_VELOCITY)

    @jax.jit
    def update(discrete_velocities_prev):
        # (1) Prescribe the outflow BC on the right boundary
        discrete_velocities_prev = discrete_velocities_prev.at[-1, :, LEFT_VELOCITIES].set(
            discrete_velocities_prev[-2, :, LEFT_VELOCITIES]
        )

        # (2) Macroscopic velocity
        density_prev = get_density(discrete_velocities_prev)
        macroscopic_velocities_prev = get_macroscopic_velocities(
            discrete_velocities_prev, density_prev
        )

        # (3) Prescribe inflow dirichilet BC using Zou/He scheme
        macroscopic_velocities_prev = macroscopic_velocities_prev.at[0, 1:-1, :].set(
            velocity_profile[0, 1:-1, :]
        )
        density_prev = density_prev.at[0, :].set(
            (get_density(discrete_velocities_prev[0, :, PURE_VERTICAL_VELOCITIES].T)
            +
            2*
            get_density(discrete_velocities_prev[0, :, LEFT_VELOCITIES].T))/ (
                1 - macroscopic_velocities_prev[0, :, 0]
            )
        )

        # (4) Compute discrete Equilibria velocities
        equilibrium_discrete_velocities  = get_equilibrium_discrete_velocities(
            macroscopic_velocities_prev, density_prev
        )
        
        # （3） Belongs to the Zou/He scheme 
        discrete_velocities_prev = discrete_velocities_prev.at[0, :, RIGHT_VELOCITIES].set(
            equilibrium_discrete_velocities[0, :, RIGHT_VELOCITIES]
        )

        # (5) Collide according to BGK
        discrete_velocities_post_collision = (
            discrete_velocities_prev-
            relaxation_omega *(
                discrete_velocities_prev
                -
                equilibrium_discrete_velocities
            )
        )

        # (6) Bounce back Boundary to enforce the no-slip
        for i in range(N_DISCRETE_VELOCITIES):
            discrete_velocities_post_collision = discrete_velocities_post_collision.at[obstacle_mask,
                                     LATTICE_INDICIES[i]].set(discrete_velocities_prev[
                                         obstacle_mask, OPPOSITE_LATTICE_INDICIES[i]]
                                         )
        
        # (7) Stream aloongside lattice velocities
        discrete_velocities_streamed = discrete_velocities_post_collision
        for i in range(N_DISCRETE_VELOCITIES):
            discrete_velocities_streamed = discrete_velocities_streamed.at[:, :, i].set(
                jnp.roll(
                    jnp.roll(
                        discrete_velocities_post_collision[:, :, i],
                        LATTICE_VELOCITIES[0, i],
                        axis=0
                    ),
                    LATTICE_VELOCITIES[1, i],
                    axis=-1,
                )
            )
        
        return discrete_velocities_streamed

    
    discrete_velocities_prev = get_equilibrium_discrete_velocities(
        velocity_profile,
        jnp.ones((N_POINTS_X, N_POINTS_Y))
    )
    
    plt.style.use("dark_background")
    plt.figure(figsize=(15, 6), dpi=100)

    
    for iteration_index in tqdm(range(N_ITERATIONS)):

        discrete_velocities_next = update(discrete_velocities_prev)
        
        discrete_velocities_prev = discrete_velocities_next

        if iteration_index % PLOT_EVERY_N_STEPS == 0 and VISUALIZE and iteration_index > SKIP_FIRST_N_ITERATIONS:
            density = get_density(discrete_velocities_next)
            macroscopic_velocities = get_macroscopic_velocities(discrete_velocities_next, density)
            velocity_magnitude = jnp.linalg.norm(macroscopic_velocities, axis=-1, ord=2)
            d_u__d_x, d_u__d_y = jnp.gradient(macroscopic_velocities[..., 0])
            d_v__d_x, d_v__d_y = jnp.gradient(macroscopic_velocities[..., 1])
            curl = (d_u__d_y - d_v__d_x)

            # Velocity Magnitude Contour Plot in the top
            plt.subplot(211)
            plt.contourf(
                X,
                Y,
                velocity_magnitude,
                levels=50,
                cmap=cmr.amber,
            )
            plt.colorbar().set_label("Velocity Magnitude")
            plt.gca().add_patch(plt.Circle(
                (CYLINDER_CENTER_X, CYLINDER_CENTER_Y),
                CYLINDER_RADIUS_INDICES,
                color="darkgreen",
            ))

            # Vorticity Magnitude Contour PLot in the bottom
            plt.subplot(212)
            plt.contourf(
                X,
                Y, 
                curl,
                levels=50,
                cmap=cmr.redshift,
                vmin=-0.02,
                vmax= 0.02,
            )
            plt.colorbar().set_label("Vorticity Magnitude")
            plt.gca().add_patch(plt.Circle(
                (CYLINDER_CENTER_X, CYLINDER_CENTER_Y),
                CYLINDER_RADIUS_INDICES,
                color="darkgreen",
            ))

            plt.draw()
            plt.pause(0.1)
            plt.clf()
    
    if VISUALIZE:
        plt.show()
    


if __name__ == "__main__":
    main()