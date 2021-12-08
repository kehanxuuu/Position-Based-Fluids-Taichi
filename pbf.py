# Macklin, M. and Müller, M., 2013. Position based fluids. ACM Transactions on Graphics (TOG), 32(4), p.104.
# 3D Taichi implementation by Jiahong, Fengshi, and Kehan
# Build upon the 2D Taichi implementation by Ye Kuang (k-ye)

import numpy as np

from utils import *


@ti.data_oriented
class Fluid(object):
    def __init__(self):
        self.old_positions = ti.Vector.field(dim, float)
        self.positions = ti.Vector.field(dim, float)
        self.velocities = ti.Vector.field(dim, float)
        self.forces = ti.Vector.field(dim, float)
        self.omegas = ti.Vector.field(dim, float)
        self.velocities_delta = ti.Vector.field(dim, float)
        self.density = ti.field(float)
        self.grid_num_particles = ti.field(int)
        self.grid2particles = ti.field(int)
        self.particle_num_neighbors = ti.field(int)
        self.particle_neighbors = ti.field(int)
        self.lambdas = ti.field(float)
        self.position_deltas = ti.Vector.field(dim, float)
        # 0: x-pos, 1: timestep in sin()
        self.board_states = ti.Vector.field(2, float)
        ti.root.dense(ti.i, num_particles).place(self.old_positions, self.positions, self.velocities, self.forces, self.omegas, self.density)
        grid_snode = ti.root.dense(ti.ijk, grid_size)
        grid_snode.place(self.grid_num_particles)
        grid_snode.dense(ti.l, max_num_particles_per_cell).place(self.grid2particles)
        nb_node = ti.root.dense(ti.i, num_particles)
        nb_node.place(self.particle_num_neighbors)
        nb_node.dense(ti.j, max_num_neighbors).place(self.particle_neighbors)
        ti.root.dense(ti.i, num_particles).place(self.lambdas, self.position_deltas, self.velocities_delta)
        ti.root.place(self.board_states)

    @ti.func
    def confine_position_to_boundary(self, p):
        bmin = particle_radius_in_world
        bmax = ti.Vector([self.board_states[None][0], boundary[1], boundary[2]]) - particle_radius_in_world
        for i in ti.static(range(dim)):
            # Use randomness to prevent particles from sticking into each other after clamping
            if p[i] <= bmin:
                p[i] = bmin + epsilon * ti.random()
            elif bmax[i] <= p[i]:
                p[i] = bmax[i] - epsilon * ti.random()
        return p

    @ti.kernel
    def move_board(self):
        # probably more accurate to exert force on particles according to hooke's law.
        b = self.board_states[None]
        b[1] += 1.0
        period = 90
        vel_strength = 8.0
        if b[1] >= 2 * period:
            b[1] = 0
        b[0] += -ti.sin(b[1] * np.pi / period) * vel_strength * time_delta
        self.board_states[None] = b

    @ti.kernel
    def find_neighbour(self):
        # clear neighbor lookup table
        for I in ti.grouped(self.grid_num_particles):
            self.grid_num_particles[I] = 0
        for I in ti.grouped(self.particle_neighbors):
            self.particle_neighbors[I] = -1

        # update grid
        for p_i in self.positions:
            cell = get_cell(self.positions[p_i])
            # ti.Vector doesn't seem to support unpacking yet
            # but we can directly use int Vectors as indices
            offs = ti.atomic_add(self.grid_num_particles[cell], 1)
            self.grid2particles[cell, offs] = p_i
        # find particle neighbors
        for p_i in self.positions:
            pos_i = self.positions[p_i]
            cell = get_cell(pos_i)
            nb_i = 0
            for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
                cell_to_check = cell + offs
                if is_in_grid(cell_to_check):
                    for j in range(self.grid_num_particles[cell_to_check]):
                        p_j = self.grid2particles[cell_to_check, j]
                        if nb_i < max_num_neighbors and p_j != p_i and (
                                pos_i - self.positions[p_j]).norm() < neighbor_radius:
                            self.particle_neighbors[p_i, nb_i] = p_j
                            nb_i += 1
            self.particle_num_neighbors[p_i] = nb_i

    @ti.kernel
    def substep(self):
        # compute lambdas
        # Eq (8) ~ (11)
        for p_i in self.positions:
            pos_i = self.positions[p_i]

            grad_i = ti.Vector([0.0, 0.0, 0.0])
            sum_gradient_sqr = 0.0
            density_constraint = 0.0

            for j in range(self.particle_num_neighbors[p_i]):
                p_j = self.particle_neighbors[p_i, j]
                if p_j < 0:
                    break
                pos_ji = pos_i - self.positions[p_j]
                # according to the paper,
                # grad_j = -1/rho_0*spiky_gradient(pos_ji, h)
                # grad_i += -grad_j
                # minus sign is omitted because of square (sign does not matter)
                # rho0 is added (previously omited in the example code)
                # grad_j = spiky_gradient(pos_ji, h) (example code)
                grad_j = spiky_gradient(pos_ji, h) * mass / rho0
                grad_i += grad_j
                sum_gradient_sqr += grad_j.dot(grad_j)
                # Eq(2)
                density_constraint += poly6_value(pos_ji.norm(), h)  # mass in Eq(2) is moved to Eq(1)

            # Eq(1)
            density_constraint += poly6_value(0, h)  # self contribution
            # grad_i += spiky_gradient(pos_zero, h)

            density_constraint = (mass * density_constraint / rho0) - 1.0
            sum_gradient_sqr += grad_i.dot(grad_i)
            self.lambdas[p_i] = (-density_constraint) / (sum_gradient_sqr + lambda_epsilon)

        # compute position deltas
        # Eq(12), (14)
        for p_i in self.positions:
            pos_i = self.positions[p_i]
            lambda_i = self.lambdas[p_i]

            pos_delta_i = ti.Vector([0.0, 0.0, 0.0])
            for j in range(self.particle_num_neighbors[p_i]):
                p_j = self.particle_neighbors[p_i, j]
                if p_j < 0:
                    break
                lambda_j = self.lambdas[p_j]
                pos_ji = pos_i - self.positions[p_j]
                scorr_ij = compute_scorr(pos_ji)
                pos_delta_i += (lambda_i + lambda_j + scorr_ij) * spiky_gradient(pos_ji, h)

            # scorr_ii = compute_scorr(pos_zero)
            # pos_delta_i += (lambda_i + lambda_i + scorr_ii) * spiky_gradient(pos_zero, h)

            pos_delta_i *= mass / rho0
            self.position_deltas[p_i] = pos_delta_i
        # apply position deltas
        for i in self.positions:
            self.positions[i] += self.position_deltas[i]

    @ti.kernel
    def update_velocity_from_position(self):
        # confine to boundary
        for i in self.positions:
            pos = self.positions[i]
            self.positions[i] = self.confine_position_to_boundary(pos)
        # update velocities
        for i in self.positions:
            self.velocities[i] = (self.positions[i] - self.old_positions[i]) / time_delta

    @ti.kernel
    def compute_density(self):
        for p_i in self.positions:
            pos_i = self.positions[p_i]
            density_constraint = 0.0

            for j in range(self.particle_num_neighbors[p_i]):
                p_j = self.particle_neighbors[p_i, j]
                if p_j < 0:
                    break
                pos_ji = pos_i - self.positions[p_j]
                # Eq(2)
                density_constraint += poly6_value(pos_ji.norm(), h)  # mass in Eq(2) is moved to Eq(1)

            # Eq(1)
            density_constraint += poly6_value(0, h)  # self contribution
            self.density[p_i] = density_constraint * mass

    @ti.kernel
    def clear_forces(self):
        for i in self.forces:
            self.forces[i] *= 0.0

    @ti.kernel
    def add_gravity(self):
        # apply gravity within boundary
        G = mass * ti.Vector([0.0, 0.0, -9.8])
        for i in self.forces:
            self.forces[i] += G

    @ti.kernel
    def add_vorticity_forces(self, Vorticity_Epsilon: ti.f32):
        # Vorticity Confinement
        for i in self.positions:
            pos_i = self.positions[i]
            self.omegas[i] = pos_i * 0.0
            for j in range(self.particle_num_neighbors[i]):
                p_j = self.particle_neighbors[i, j]
                if p_j < 0:
                    break
                pos_ji = pos_i - self.positions[p_j]
                self.omegas[i] += mass * (self.velocities[p_j] - self.velocities[i]).cross(spiky_gradient(pos_ji, h)) / (epsilon + self.density[p_j])

        for i in self.positions:
            pos_i = self.positions[i]
            loc_vec_i = pos_i * 0.0
            for j in range(self.particle_num_neighbors[i]):
                p_j = self.particle_neighbors[i, j]
                if p_j < 0:
                    break
                pos_ji = pos_i - self.positions[p_j]
                loc_vec_i += mass * self.omegas[p_j].norm() * spiky_gradient(pos_ji, h) / (epsilon + self.density[p_j])
            omega_i = self.omegas[i]
            # loc_vec_i += mass * omega_i.norm() * spiky_gradient(pos_i * 0.0, h) / (epsilon + density[i])
            loc_vec_i = loc_vec_i / (epsilon + loc_vec_i.norm())
            self.forces[i] += Vorticity_Epsilon * loc_vec_i.cross(omega_i)

    @ti.kernel
    def apply_forces(self):
        for i in self.velocities:
            self.velocities[i] += self.forces[i] / mass * time_delta

    @ti.kernel
    def update_position_from_velocity(self):
        for i in self.positions:
            self.positions[i] += self.velocities[i] * time_delta
            self.positions[i] = self.confine_position_to_boundary(self.positions[i])

    @ti.kernel
    def apply_viscosity(self, XSPH_C: ti.f32):
        # XSPH Artificial Viscosity -> no obvious effect?
        for i in self.positions:
            pos_i = self.positions[i]
            self.velocities_delta[i] = pos_i * 0.0
            for j in range(self.particle_num_neighbors[i]):
                p_j = self.particle_neighbors[i, j]
                if p_j < 0:
                    break
                pos_ji = pos_i - self.positions[p_j]
                self.velocities_delta[i] += mass * (self.velocities[p_j] - self.velocities[i]) * poly6_value(pos_ji.norm(), h) / (
                        epsilon + self.density[p_j])  # (velocities[p_j] - velocities[i]) * poly6_value(pos_ji.norm(), h)

        for i in self.positions:
            self.velocities[i] += XSPH_C * self.velocities_delta[i]

    @ti.kernel
    def save_old_pos(self):
        for i in self.positions:
            self.old_positions[i] = self.positions[i]

    def run_pbf(self):
        # method 1: our way of updating, 0.06s per frame on mac
        # find neighbour twice (time consuming)
        # save_old_pos()
        # clear_forces()
        # add_gravity()
        # # voricity confinement
        # find_neighbour()
        # compute_density()
        # add_vorticity_forces(vorticity_epsilon)
        # # TODO: damping
        # apply_forces()
        # apply_viscosity(xsph_c)
        # update_position_from_velocity()

        # # PBD Algorithm:
        # find_neighbour()
        # for _ in range(pbf_num_iters):
        #     substep()
        # update_velocity_from_position()

        # method 2: same to the paper, 0.046s per frame on mac
        # only find neighbour once, but clear and apply force twice
        self.save_old_pos()
        self.clear_forces()
        self.add_gravity()
        self.apply_forces()
        self.update_position_from_velocity()

        # PBD Algorithm:
        self.find_neighbour()
        for _ in range(pbf_num_iters):
            self.substep()
        self.update_velocity_from_position()
        self.clear_forces()
        self.compute_density()
        self.add_vorticity_forces(vorticity_epsilon)
        self.apply_forces()
        self.apply_viscosity(xsph_c)

    @ti.kernel
    def init_particles(self):
        for i in range(num_particles):
            delta = h * 0.8
            num_particles_xy = num_particles_x * num_particles_y
            i_mod_xy = i % num_particles_xy
            i_mod_x = i % num_particles_x
            offs = ti.Vector([(boundary[0] - delta * num_particles_x) * (0.0 if i_mod_x < num_particles_x // 2 else 0.9),
                              (boundary[1] - delta * num_particles_y) * 0.5,
                              boundary[2] * 0.5])
            self.positions[i] = ti.Vector([i_mod_x, i_mod_xy // num_particles_x, i // num_particles_xy]) * delta + offs
            for c in ti.static(range(dim)):
                self.velocities[i][c] = (ti.random() - 0.5) * 4
        self.board_states[None] = ti.Vector([boundary[0] - epsilon, -0.0])

    def print_stats(self, time_interval):
        print('PBF stats:')
        num = self.grid_num_particles.to_numpy()
        avg, max = np.mean(num), np.max(num)
        print(f'  #particles per cell: avg={avg:.2f} max={max}')
        num = self.particle_num_neighbors.to_numpy()
        avg, max = np.mean(num), np.max(num)
        print(f'  #neighbors per particle: avg={avg:.2f} max={max}')
        print(f'  #vorticity_epsilon value: {vorticity_epsilon:.5f}')
        print(f'  #xsph_c value: {xsph_c:.5f}')
        print(f'  #time per frame: {time_interval:.5f}')
