# Macklin, M. and Müller, M., 2013. Position based fluids. ACM Transactions on Graphics (TOG), 32(4), p.104.
# 3D Taichi implementation by Jiahong, Fengshi, and Kehan
# Build upon the 2D Taichi implementation by Ye Kuang (k-ye)

import numpy as np
import open3d as o3d
from matplotlib import cm

from utils import *

from rigidbody import RigidObjectField


@ti.data_oriented
class Fluid(object):
    def __init__(self):
        self.dt = time_delta
        self.pbf_num_iters = pbf_num_iters
        self.xsph_c = xsph_c
        self.vorticity_epsilon = vorticity_epsilon
        self.collisions_eps = collision_eps
        self.particle_color = particle_color

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

        self.pcd = o3d.geometry.PointCloud()

        self.boundary_handled_by_collision = ti.field(int, shape=())
        self.boundary_handled_by_confinement = ti.field(int, shape=())

    @ti.kernel
    def clear_stats(self):
        self.boundary_handled_by_collision[None] = 0
        self.boundary_handled_by_confinement[None] = 0

    @ti.func
    def confine_position_to_boundary(self, p):
        """
        Simply confine the position into the boundary, no collision response generated
        """
        bmin = particle_radius_in_world * 0.1
        bmax = ti.Vector([self.board_states[None][0], boundary[1], boundary[2]]) - particle_radius_in_world * 0.1
        has_confinement = 0
        for i in ti.static(range(dim)):
            # Use randomness to prevent particles from sticking into each other after clamping
            if p[i] <= bmin:
                p[i] = bmin + epsilon * ti.random()
                has_confinement = 1
            elif bmax[i] <= p[i]:
                p[i] = bmax[i] - epsilon * ti.random()
                has_confinement = 1
        if has_confinement >= 0.5:
            self.boundary_handled_by_confinement[None] += 1
        return p

    @ti.kernel
    def handle_boundary_collisions(self, eps: ti.f32):
        """
        Detect boundary collisions, confine positions, and add impulse to velocities
        """
        bmin = particle_radius_in_world
        bmax = ti.Vector([self.board_states[None][0], boundary[1], boundary[2]]) - particle_radius_in_world
        for i in self.positions:
            pos = self.positions[i]
            has_collision = 0
            normal = ti.Vector([0.0, 0.0, 0.0])  # from solid to liquid
            v_boundary = ti.Vector([0.0, 0.0, 0.0])
            for j in ti.static(range(dim)):
                # Use randomness to prevent particles from sticking into each other after clamping
                if pos[j] <= bmin:

                    pos[j] = bmin + epsilon * ti.random()
                    normal[j] = 1.0
                    has_collision = 1
                elif bmax[j] <= pos[j]:
                    pos[j] = bmax[j] - epsilon * ti.random()
                    normal[j] = -1.0
                    has_collision = 1
                    if j == 0:  # hit the board
                        v_boundary[j] = self.board_speed(self.board_states[None][1])

            self.positions[i] = pos
            if has_collision >= 0.5:
                # add impulse from boundary to velocity
                self.boundary_handled_by_collision[None] += 1
                normal = normal.normalized()
                vel = self.velocities[i]
                vrel_before = vel - v_boundary
                vrel_before_orth = vrel_before.dot(normal) * normal
                vrel_before_para = vrel_before - vrel_before_orth
                vrel_after = vrel_before_para + -eps * vrel_before_orth
                self.velocities[i] = vrel_after + v_boundary

    @ti.kernel
    def move_board(self):
        # probably more accurate to exert force on particles according to hooke's law.
        b = self.board_states[None]
        b[1] += 1.0
        period = 90
        if b[1] >= 2 * period:
            b[1] = 0
        b[0] += self.board_speed(b[1]) * self.dt
        self.board_states[None] = b

    @ti.func
    def board_speed(self, t: ti.f32) -> ti.f32:
        period = 90
        vel_strength = 8.0
        return -ti.sin(t * np.pi / period) * vel_strength

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
            self.velocities[i] = (self.positions[i] - self.old_positions[i]) / self.dt

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
        G = mass * ti.Vector([0.0, 0.0, -g_const])
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
            self.velocities[i] += self.forces[i] / mass * self.dt

    @ti.kernel
    def update_position_from_velocity(self):
        for i in self.positions:
            self.positions[i] += self.velocities[i] * self.dt
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
        self.handle_boundary_collisions(self.collisions_eps)  # regard collision impulse as external forces
        self.update_position_from_velocity()

        # PBD Algorithm:
        self.find_neighbour()
        for _ in range(self.pbf_num_iters):
            self.substep()
        self.update_velocity_from_position()

        self.clear_forces()
        self.compute_density()
        self.add_vorticity_forces(self.vorticity_epsilon)
        self.apply_forces()
        self.apply_viscosity(self.xsph_c)

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
        print(f'  #fps: {1 / time_interval:.2f}')
        print(f'  #vorticity_epsilon value: {self.vorticity_epsilon:.5f}')
        print(f'  #xsph_c value: {self.xsph_c:.5f}')
        print(f'  #collisionEps value: {self.collisions_eps:.5f}')
        density_np = self.density.to_numpy()
        print(f'  {density_np.min()=}, {density_np.max()=}, {density_np.mean()=}')
        collision = self.boundary_handled_by_collision.to_numpy()
        confinement = self.boundary_handled_by_confinement.to_numpy()
        total_boundary = collision + confinement
        print(f'  boundary handled by collision:   {collision / float(total_boundary) * 100:.2f}%')
        print(f'  boundary handled by confinement: {confinement / float(total_boundary) * 100:.2f}%')
        self.clear_stats()

    def update_point_cloud(self):
        pos_np = self.positions.to_numpy()
        pos_np *= screen_to_world_ratio
        pos_np = pos_np[:, (0, 2, 1)]  # recap: z and y axis in the simulation are swapped for better visualization
        self.pcd.points = o3d.utility.Vector3dVector(pos_np)
        if self.particle_color == 'velocity':
            velnorm_np = np.linalg.norm(self.velocities.to_numpy(), axis=1) / cm_max_velocity
            self.pcd.colors = o3d.utility.Vector3dVector(cm.jet(velnorm_np)[:, :3])
        elif self.particle_color == 'density':
            # fluid.find_neighbour()
            # fluid.compute_density()
            density_np = self.density.to_numpy()
            density_np = density_np / rho0 * 0.5  # map to [0, 1]
            self.pcd.colors = o3d.utility.Vector3dVector(cm.RdBu(density_np)[:, :3])
        elif self.particle_color == 'vorticity':
            omegas_np = np.linalg.norm(self.omegas.to_numpy(), axis=1) / cm_max_vorticity
            self.pcd.colors = o3d.utility.Vector3dVector(cm.YlGnBu(omegas_np)[:, :3])
        else:
            raise ValueError(f'Unknown particle color key {self.particle_color}')
