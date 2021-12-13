import numpy as np
import json
import subprocess
import os

# perlin noise
# usage: generate_heighmap_noise(terrain_width, terrain_height, scale)
def lerp_normal(a, t):
    return (1.0 - t) * a[0] + t * a[1]

def hash(p):
	x = np.array([np.dot(p, [127.1, 311.7]), np.dot(p, [269.5,183.3]) ])
	x = np.sin(x) * 43758.5453123
	x_floor = np.floor(x)
	x_frac = x - x_floor
	return -1.0 + 2.0 * x_frac # [-1, 1]
	
def perlin_noise(p):
	p_floor = np.floor(p)
	p_frac = p - p_floor
	smooth_interval = p_frac * p_frac * (3.0 - 2.0 * p_frac)
	x_1 = lerp_normal([np.dot(hash(p_floor + np.array([0.0, 0.0])), p_frac - np.array([0.0, 0.0])),
	      		np.dot(hash(p_floor + np.array([1.0, 0.0])), p_frac - np.array([1.0, 0.0])),
	     	   ], smooth_interval[0])
	x_2 = lerp_normal([np.dot(hash(p_floor + np.array([0.0, 1.0])), p_frac - np.array([0.0, 1.0])),
	      		np.dot(hash(p_floor + np.array([1.0, 1.0])), p_frac - np.array([1.0, 1.0])),
	     	   ], smooth_interval[0])
	return lerp_normal([x_1, x_2], smooth_interval[1])
	
def fractional_noise(p):
	f = 0.0
	p = p * 4.0
	f += 1.0000 * perlin_noise(p)
	p = 2.0 * p
	f += 0.5000 * perlin_noise(p)
	p = 2.0 * p
	f += 0.2500 * perlin_noise(p)
	p = 2.0 * p
	f += 0.1250 * perlin_noise(p)
	p = 2.0 * p
	f += 0.0625 * perlin_noise(p)
	
	return f
	
def generate_heighmap_noise(width, height, scale=1, ratioX=3.1, ratioY=1.3, shift=[0, 0]):
	noise = np.zeros([width, height])
	for i in range(0, width):
		for j in range(0, height):
            # avoid (0, 0), where noise is definitely 0
			noise[i][j] = fractional_noise(np.array([(i+1)*ratioX + shift[0], (j+1)*ratioY] + shift[1]))
	return noise * scale


# particle to mesh
def convert_particle_info_to_json(input, filepath):
    filepath = filepath + ".json"
    input_list = input.tolist()
    with open(filepath, 'w') as outfile:
        json.dump(input_list, outfile)

def convert_rigid_info_to_json(input, filepath):
	# input is a dict
    filepath = filepath + ".json"
    with open(filepath, 'w') as outfile:
        json.dump(input, outfile)

def convert_json_to_mesh_command_line(particle_dir, mesh_dir, filename,
                                      particle_radius=0.8,
                                      smoothing_length=2.0,
                                      cube_size=0.5,
                                      surface_threshold=0.6
                                     ):
    # need to install rust tool chain & splashsurf: https://github.com/w1th0utnam3/splashsurf
    filepath_particle = os.path.join(particle_dir, filename) + ".json"
    filename_mesh = filename + ".obj"
    # todo: splashsurf supports batch processing, but only for .vtk and not for .obj
    bashCommand = "splashsurf reconstruct -i {} --output-dir={} -o {} --particle-radius={} --smoothing-length={} --cube-size={} --surface-threshold={} --normals=on".format(filepath_particle, mesh_dir, filename_mesh, particle_radius, smoothing_length, cube_size, surface_threshold)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    _, _ = process.communicate() # output, error


def clear_directory(directory):
    for filename in os.listdir(directory):
        os.remove(os.path.join(directory, filename))