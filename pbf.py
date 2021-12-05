# Macklin, M. and Müller, M., 2013. Position based fluids. ACM Transactions on Graphics (TOG), 32(4), p.104.
# 3D Taichi implementation by Jiahong, Fengshi, and Kehan
# Build upon the 2D Taichi implementation by Ye Kuang (k-ye)

import math

import numpy as np

import taichi as ti
import open3d as o3d

import matplotlib.cm as cm

ti.init(arch=ti.cpu)

screen_res = (800, 400, 400)  # z and y axis in the simulation are swapped for better visualization
screen_to_world_ratio = 10.0
boundary = (screen_res[0] / screen_to_world_ratio,
            screen_res[2] / screen_to_world_ratio,
            screen_res[1] / screen_to_world_ratio)
cell_size = 2.51
cell_recpr = 1.0 / cell_size

cm_max_velocity = 15
cm_max_vorticity = 20

def round_up(f, s):
    return (math.floor(f * cell_recpr / s) + 1) * s


grid_size = (round_up(boundary[0], 1), round_up(boundary[1], 1), round_up(boundary[2], 1))

dim = 3
bg_color = 0x112f41
particle_color = 'density'
boundary_color = 0xebaca2
num_particles_x = 30
num_particles_y = 20
num_particles_z = 20
num_particles = num_particles_x * num_particles_y * num_particles_z
max_num_particles_per_cell = 100
max_num_neighbors = 100
time_delta = 1.0 / 20.0
epsilon = 1e-5
particle_radius = 1.2  # change from 3.0
particle_radius_in_world = particle_radius / screen_to_world_ratio

# PBF params
h = 1.1
mass = 1.0
rho0 = 1.0
lambda_epsilon = 100.0
pbf_num_iters = 5
corr_deltaQ_coeff = 0.3  # in the paper, 0.1-0.3 for use in Eq(13)
corrK = 0.001
# Need ti.pow()
# corrN = 4.0
neighbor_radius = h * 1.05  # TODO: need to change

poly6_factor = 315.0 / 64.0 / math.pi
spiky_grad_factor = -45.0 / math.pi

vorticity_epsilon = 0.1
xsph_c = 0.00

old_positions = ti.Vector.field(dim, float)
positions = ti.Vector.field(dim, float)
velocities = ti.Vector.field(dim, float)
forces = ti.Vector.field(dim, float)
omegas = ti.Vector.field(dim, float)
grid_num_particles = ti.field(int)
grid2particles = ti.field(int)
particle_num_neighbors = ti.field(int)
particle_neighbors = ti.field(int)
lambdas = ti.field(float)
position_deltas = ti.Vector.field(dim, float)
# 0: x-pos, 1: timestep in sin()
board_states = ti.Vector.field(2, float)

ti.root.dense(ti.i, num_particles).place(old_positions, positions, velocities, forces, omegas)
grid_snode = ti.root.dense(ti.ijk, grid_size)
grid_snode.place(grid_num_particles)
grid_snode.dense(ti.l, max_num_particles_per_cell).place(grid2particles)
nb_node = ti.root.dense(ti.i, num_particles)
nb_node.place(particle_num_neighbors)
nb_node.dense(ti.j, max_num_neighbors).place(particle_neighbors)
ti.root.dense(ti.i, num_particles).place(lambdas, position_deltas)
ti.root.place(board_states)

if particle_color == 'density':
    density = ti.field(float)
    ti.root.dense(ti.i, num_particles).place(density)

@ti.func
def poly6_value(s, h):
    result = 0.0
    if 0 <= s and s < h:
        x = (h * h - s * s) / (h * h * h)
        result = poly6_factor * x * x * x
    return result


@ti.func
def spiky_gradient(r, h):
    result = ti.Vector([0.0, 0.0, 0.0])
    r_len = r.norm()
    if 0 < r_len and r_len < h:
        x = (h - r_len) / (h * h * h)
        g_factor = spiky_grad_factor * x * x
        result = r * g_factor / r_len
    return result


@ti.func
def compute_scorr(pos_ji):
    # Eq (13)
    x = poly6_value(pos_ji.norm(), h) / poly6_value(corr_deltaQ_coeff * h, h)
    # pow(x, 4)
    x = x * x
    x = x * x
    return (-corrK) * x


@ti.func
def get_cell(pos):
    return int(pos * cell_recpr)


@ti.func
def is_in_grid(c):
    # @c: Vector(i32)
    return 0 <= c[0] and c[0] < grid_size[0] and 0 <= c[1] and c[1] < grid_size[1] and 0 <= c[2] and c[2] < grid_size[2]


@ti.func
def confine_position_to_boundary(p):
    bmin = particle_radius_in_world
    bmax = ti.Vector([board_states[None][0], boundary[1], boundary[2]]) - particle_radius_in_world
    for i in ti.static(range(dim)):
        # Use randomness to prevent particles from sticking into each other after clamping
        if p[i] <= bmin:
            p[i] = bmin + epsilon * ti.random()
        elif bmax[i] <= p[i]:
            p[i] = bmax[i] - epsilon * ti.random()
    return p


@ti.kernel
def move_board():
    # probably more accurate to exert force on particles according to hooke's law.
    b = board_states[None]
    b[1] += 1.0
    period = 90
    vel_strength = 8.0
    if b[1] >= 2 * period:
        b[1] = 0
    b[0] += -ti.sin(b[1] * np.pi / period) * vel_strength * time_delta
    board_states[None] = b


@ti.kernel
def prologue():
    # clear neighbor lookup table
    for I in ti.grouped(grid_num_particles):
        grid_num_particles[I] = 0
    for I in ti.grouped(particle_neighbors):
        particle_neighbors[I] = -1

    # update grid
    for p_i in positions:
        cell = get_cell(positions[p_i])
        # ti.Vector doesn't seem to support unpacking yet
        # but we can directly use int Vectors as indices
        offs = ti.atomic_add(grid_num_particles[cell], 1)
        grid2particles[cell, offs] = p_i
    # find particle neighbors
    for p_i in positions:
        pos_i = positions[p_i]
        cell = get_cell(pos_i)
        nb_i = 0
        for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
            cell_to_check = cell + offs
            if is_in_grid(cell_to_check):
                for j in range(grid_num_particles[cell_to_check]):
                    p_j = grid2particles[cell_to_check, j]
                    if nb_i < max_num_neighbors and p_j != p_i and (
                            pos_i - positions[p_j]).norm() < neighbor_radius:
                        particle_neighbors[p_i, nb_i] = p_j
                        nb_i += 1
        particle_num_neighbors[p_i] = nb_i


@ti.kernel
def substep():
    # compute lambdas
    # Eq (8) ~ (11)
    
    pos_zero = 0.0 * positions[0]
    for p_i in positions:
        pos_i = positions[p_i]

        grad_i = ti.Vector([0.0, 0.0, 0.0])
        sum_gradient_sqr = 0.0
        density_constraint = 0.0

        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            if p_j < 0:
                break
            pos_ji = pos_i - positions[p_j]
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
        grad_i += spiky_gradient(pos_zero, h)

        density_constraint = (mass * density_constraint / rho0) - 1.0
        sum_gradient_sqr += grad_i.dot(grad_i)
        lambdas[p_i] = (-density_constraint) / (sum_gradient_sqr + lambda_epsilon)
    
    # compute position deltas
    # Eq(12), (14)
    for p_i in positions:
        pos_i = positions[p_i]
        lambda_i = lambdas[p_i]

        pos_delta_i = ti.Vector([0.0, 0.0, 0.0])
        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            if p_j < 0:
                break
            lambda_j = lambdas[p_j]
            pos_ji = pos_i - positions[p_j]
            scorr_ij = compute_scorr(pos_ji)
            pos_delta_i += (lambda_i + lambda_j + scorr_ij) * spiky_gradient(pos_ji, h)

        scorr_ii = compute_scorr(pos_zero)
        pos_delta_i += (lambda_i + lambda_i + scorr_ii) * spiky_gradient(pos_zero, h)  # self contribution

        pos_delta_i *= mass / rho0
        position_deltas[p_i] = pos_delta_i
    # apply position deltas
    for i in positions:
        positions[i] += position_deltas[i]


@ti.kernel
def epilogue():
    # confine to boundary
    for i in positions:
        pos = positions[i]
        positions[i] = confine_position_to_boundary(pos)
    # update velocities
    for i in positions:
        velocities[i] = (positions[i] - old_positions[i]) / time_delta

    # no need to update neighbour particle list regardless of change in positions, just as in multiple iterations of substep

@ti.kernel
def compute_density():
    for p_i in positions:
        pos_i = positions[p_i]
        density_constraint = 0.0

        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            if p_j < 0:
                break
            pos_ji = pos_i - positions[p_j]
            # Eq(2)
            density_constraint += poly6_value(pos_ji.norm(), h)  # mass in Eq(2) is moved to Eq(1)

        # Eq(1)
        density_constraint += poly6_value(0, h)  # self contribution
        density[p_i] = mass * density_constraint

        
@ti.kernel
def clear_forces():
    for i in forces:
        forces[i] *= 0.0


@ti.kernel
def add_gravity():
    # apply gravity within boundary
    G = mass * ti.Vector([0.0, 0.0, -9.8])
    for i in forces:
        forces[i] += G


@ti.kernel
def add_vorticity_forces(Vorticity_Epsilon: ti.f32):
    # Vorticity Confinement
    for i in positions:
        pos_i = positions[i]
        omegas[i] = pos_i * 0.0
        for j in range(particle_num_neighbors[i]):
            p_j = particle_neighbors[i, j]
            if p_j < 0:
                break
            pos_ji = pos_i - positions[p_j]
            omegas[i] += mass * (velocities[j] - velocities[i]).cross(spiky_gradient(pos_ji, h)) / rho0

    for i in positions:
        pos_i = positions[i]
        loc_vec_i = pos_i * 0.0
        for j in range(particle_num_neighbors[i]):
            p_j = particle_neighbors[i, j]
            if p_j < 0:
                break
            pos_ji = pos_i - positions[p_j]
            loc_vec_i += mass * omegas[j].norm() * spiky_gradient(pos_ji, h) / rho0
        omega_i = omegas[i]
        loc_vec_i += mass * omega_i.norm() * spiky_gradient(pos_i * 0.0, h) / rho0
        loc_vec_i = loc_vec_i / (epsilon + loc_vec_i.norm())
        forces[i] += Vorticity_Epsilon * loc_vec_i.cross(omega_i)


@ti.kernel
def apply_forces():
    for i in velocities:
        velocities[i] += forces[i] / mass * time_delta


@ti.kernel
def update_positions():
    for i in positions:
        positions[i] += velocities[i] * time_delta
        positions[i] = confine_position_to_boundary(positions[i])


@ti.kernel
def apply_viscosity(XSPH_C: ti.f32):
    # XSPH Artificial Viscosity -> no obvious effect?
    for i in positions:
        pos_i = positions[i]
        velocity_delta_i = pos_i * 0.0
        for j in range(particle_num_neighbors[i]):
            p_j = particle_neighbors[i, j]
            if p_j < 0:
                break
            pos_ji = pos_i - positions[p_j]
            velocity_delta_i += mass * (velocities[j] - velocities[i]) * poly6_value(pos_ji.norm(), h) / rho0
        velocities[i] += XSPH_C * velocity_delta_i


@ti.kernel
def save_old_pos():
    for i in positions:
        old_positions[i] = positions[i]


def run_pbf():
    save_old_pos()
    clear_forces()
    add_gravity()
    add_vorticity_forces(vorticity_epsilon)
    # TODO: damping
    apply_forces()
    apply_viscosity(xsph_c)
    update_positions()

    # PBD Algorithm:
    prologue()
    for _ in range(pbf_num_iters):
        substep()
    epilogue()


def render(vis, pcd, box):
    pos_np = positions.to_numpy()
    pos_np *= screen_to_world_ratio
    pos_np = pos_np[:, (0, 2, 1)]  # recap: z and y axis in the simulation are swapped for better visualization
    pcd.points = o3d.utility.Vector3dVector(pos_np)
    if particle_color == 'velocity':
        velnorm_np = np.linalg.norm(velocities.to_numpy(), axis=1) / cm_max_velocity
        pcd.colors = o3d.utility.Vector3dVector(cm.jet(velnorm_np)[:, :3])
    elif particle_color == 'density':
        prologue()
        compute_density()
        density_np = density.to_numpy()
        if (density_np<rho0).any():
            print('Exists density smaller than rho0')
        density_np = (density_np - rho0) * 0.5 + 0.5
        pcd.colors = o3d.utility.Vector3dVector(cm.RdBu(density_np)[:, :3])
    elif particle_color == 'vorticity':
        omegas_np = np.linalg.norm(omegas.to_numpy(), axis=1) / cm_max_vorticity
        pcd.colors = o3d.utility.Vector3dVector(cm.YlGnBu(omegas_np)[:, :3])
    else:
        raise 'unknown particle color key'
    vis.update_geometry(pcd)
    box.translate(np.array([board_states[None][0], boundary[1] / 2, boundary[2] / 2]) * screen_to_world_ratio, relative=False)
    vis.update_geometry(box)


@ti.kernel
def init_particles():
    for i in range(num_particles):
        delta = h * 0.8
        num_particles_xy = num_particles_x * num_particles_y
        i_mod_xy = i % num_particles_xy
        i_mod_x = i % num_particles_x
        offs = ti.Vector([(boundary[0] - delta * num_particles_x) * (0.0 if i_mod_x < num_particles_x // 2 else 0.9),
                          (boundary[1] - delta * num_particles_y) * 0.5,
                          boundary[2] * 0.5])
        positions[i] = ti.Vector([i_mod_x, i_mod_xy // num_particles_x, i // num_particles_xy]) * delta + offs
        for c in ti.static(range(dim)):
            velocities[i][c] = (ti.random() - 0.5) * 4
    board_states[None] = ti.Vector([boundary[0] - epsilon, -0.0])


def print_stats():
    print('PBF stats:')
    num = grid_num_particles.to_numpy()
    avg, max = np.mean(num), np.max(num)
    print(f'  #particles per cell: avg={avg:.2f} max={max}')
    num = particle_num_neighbors.to_numpy()
    avg, max = np.mean(num), np.max(num)
    print(f'  #neighbors per particle: avg={avg:.2f} max={max}')
    print(f'  #vorticity_epsilon value: {vorticity_epsilon:.5f}')
    print(f'  #xsph_c value: {xsph_c:.5f}')


reset = False
paused = True


def main():
    # setup gui
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    def resetSimulation(vis):
        global reset
        reset = not reset

    def pause(vis):
        global paused
        paused = not paused

    def increaseVorticity(vis):
        global vorticity_epsilon
        vorticity_epsilon += 0.05

    def decreaseVorticity(vis):
        global vorticity_epsilon
        vorticity_epsilon -= 0.05
        if vorticity_epsilon <= 0:
            vorticity_epsilon = 0

    def increaseViscosity(vis):
        global xsph_c
        xsph_c += 0.01

    def decreaseViscosity(vis):
        global xsph_c
        xsph_c -= 0.01
        if xsph_c <= 0:
            xsph_c = 0

    vis.register_key_callback(ord("R"), resetSimulation)
    vis.register_key_callback(ord(" "), pause)  # space
    vis.register_key_callback(ord("E"), increaseVorticity)
    vis.register_key_callback(ord("D"), decreaseVorticity)
    vis.register_key_callback(ord("W"), increaseViscosity)
    vis.register_key_callback(ord("S"), decreaseViscosity)

    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=100, origin=[0, 0, 0]
    )
    vis.add_geometry(coordinate_frame)  # coordinate frame

    aabb = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=np.array([0, 0, 0]), max_bound=np.array(screen_res)
    )
    aabb.color = [0.7, 0.7, 0.7]
    vis.add_geometry(aabb)  # bounding box

    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    box = o3d.geometry.TriangleMesh.create_box(5, screen_res[1], screen_res[2])
    box.translate(np.array([screen_res[0], 0, 0]))
    vis.add_geometry(box)

    init_particles()
    print(f'boundary={boundary} grid={grid_size} cell_size={cell_size}')

    iter = 0
    while True:
        if reset:
            init_particles()
            resetSimulation(vis)
            print(f'reset simulation')

        if not paused:
            move_board()
            run_pbf()
            if iter % 20 == 1:
                print_stats()
            iter += 1
            render(vis, pcd, box)

        if not vis.poll_events():
            break

        if not paused:
            vis.update_renderer()


if __name__ == '__main__':
    main()
