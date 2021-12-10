import numpy as np

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