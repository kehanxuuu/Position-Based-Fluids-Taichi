import math

device = 'gpu'

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
num_particles_x = 20
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
g_const = 9.8
lambda_epsilon = 100.0
pbf_num_iters = 5
corr_deltaQ_coeff = 0.3  # in the paper, 0.1-0.3 for use in Eq(13)
corrK = 0.001
# Need ti.pow()
# corrN = 4.0
neighbor_radius = h * 1.05  # TODO: need to change

poly6_factor = 315.0 / 64.0 / math.pi
spiky_grad_factor = -45.0 / math.pi

vorticity_epsilon = 0.0
xsph_c = 0.0