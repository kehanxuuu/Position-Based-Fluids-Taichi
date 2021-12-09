import taichi as ti

from config import *


@ti.func
def lerp(t, a, b):
    return (1.0 - t) * a + t * b


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
def get_cell(pos):
    return int(pos * cell_recpr)


@ti.func
def compute_scorr(pos_ji):
    # Eq (13)
    x = poly6_value(pos_ji.norm(), h) / poly6_value(corr_deltaQ_coeff * h, h)
    # pow(x, 4)
    x = x * x
    x = x * x
    return (-corrK) * x


@ti.func
def is_in_grid(c):
    # @c: Vector(i32)
    return 0 <= c[0] and c[0] < grid_size[0] and 0 <= c[1] and c[1] < grid_size[1] and 0 <= c[2] and c[2] < grid_size[2]


###### QUATERNIONS ######
# Quaternions are represented with taichi 4-vectors
# with the format q = ti.Vector([x, y, z, w]) = w + ix + jy + kz


@ti.func
def vector_to_quat(v):
    return ti.Vector([v.x, v.y, v.z, 0])


@ti.func
def quaternion_multiply(p, q):
    ret = ti.Vector(
        [
            p.x * q.w + p.w * q.x + p.y * q.z - p.z * q.y,
            p.y * q.w + p.w * q.y + p.z * q.x - p.x * q.z,
            p.z * q.w + p.w * q.z + p.x * q.y - p.y * q.x,
            p.w * q.w - p.x * q.x - p.y * q.y - p.z * q.z,
        ]
    )
    return ret


@ti.func
def quaternion_to_matrix(q):
    # First row of the rotation matrix
    r00 = 2 * (q.w * q.w + q.x * q.x) - 1
    r01 = 2 * (q.x * q.y - q.w * q.z)
    r02 = 2 * (q.x * q.z + q.w * q.y)

    # Second row of the rotation matrix
    r10 = 2 * (q.x * q.y + q.w * q.z)
    r11 = 2 * (q.w * q.w + q.y * q.y) - 1
    r12 = 2 * (q.y * q.z - q.w * q.x)

    # Third row of the rotation matrix
    r20 = 2 * (q.x * q.z - q.w * q.y)
    r21 = 2 * (q.y * q.z + q.w * q.x)
    r22 = 2 * (q.w * q.w + q.z * q.z) - 1

    # 3x3 rotation matrix
    rot_matrix = ti.Matrix([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])
    return rot_matrix


@ti.func
def matrix_to_quaternion(M):
    qw = 0.5 * ti.sqrt(1 + M[0, 0] + M[1, 1] + M[2, 2])
    qx = (M[2, 1] - M[1, 2]) / (4 * qw)
    qy = (M[0, 2] - M[2, 0]) / (4 * qw)
    qz = (M[1, 0] - M[0, 1]) / (4 * qw)

    q = ti.Vector([qx, qy, qz, qw])
    return q


@ti.func
def quaternion_inverse(q):
    qInv = ti.Vector([-q.x, -q.y, -q.z, q.w]) / q.norm()
    return qInv