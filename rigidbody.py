import taichi as ti
import open3d as o3d
import numpy as np
import utils
import itertools
from config import time_delta, boundary, screen_to_world_ratio, g_const, grid_size, rigid_rigid_eps, epsilon


@ti.data_oriented
class ObjType:
    STATIC = 0
    DYNAMIC = 1


@ti.data_oriented
class RigidObjectField(object):
    def __init__(self, paths, scalings=1.0):
        """
        Initialize a rigid body **field** with given path and scale settings.
        If the params are a single str and float, the returned rigid body field is actually one rigid body.
        If the params are ndarrays, the returned rigid body field has the same shape as params.

        :param paths: str or ndarray of strings, path to .off mesh files
        :param scalings: float or ndarray of floats, scale factor applied to the mesh vertices
        """
        self.paths = np.array(paths)
        self.scalings = np.array(scalings, dtype=np.float32)
        self.shape = self.paths.shape
        # meshes are numpy arrays in favor of rendering
        self.meshes = np.ndarray(shape=self.shape, dtype=o3d.geometry.TriangleMesh)

        if self.scalings.shape != self.shape:
            raise AttributeError(f"Error: scalings does not have the correct shape! {self.scalings.shape} instead of {self.shape}")

        # create ranges to iterate over
        self.shape_ranges = []
        for dim in self.shape:
            self.shape_ranges.append(list(range(dim)))

        # create the meshes
        n_vertices = np.ndarray(shape=self.shape, dtype=np.int32)
        n_faces = np.ndarray(shape=self.shape, dtype=np.int32)
        self.num_meshes = 0
        for e in itertools.product(*self.shape_ranges):
            self.meshes[e] = o3d.io.read_triangle_mesh(self.paths[e])
            self.meshes[e].compute_vertex_normals()
            self.meshes[e].compute_triangle_normals()
            n_vertices[e] = len(self.meshes[e].vertices)
            n_faces[e] = len(self.meshes[e].triangles)
            self.num_meshes += 1

        self.max_num_vertices = n_vertices.max()
        self.max_num_faces = n_faces.max()
        self.sum_num_vertices = n_vertices.sum()
        self.sum_num_faces = n_faces.sum()

        # V, new_V are taichi fields for simulation
        self.nV = ti.field(dtype=ti.i32, shape=self.shape)
        self.nF = ti.field(dtype=ti.i32, shape=self.shape)
        self.V = ti.Vector.field(3, ti.f32, (*self.shape, self.max_num_vertices))
        self.new_V = ti.Vector.field(3, ti.f32, (*self.shape, self.max_num_vertices))
        self.F = ti.Vector.field(3, ti.i32, (*self.shape, self.max_num_faces))
        self.C = ti.Vector.field(3, ti.f32, (*self.shape, self.max_num_vertices))

        vertices = np.zeros(shape=(*self.shape, self.max_num_vertices, 3), dtype=np.float32)
        faces = np.zeros(shape=(*self.shape, self.max_num_faces, 3), dtype=np.int32)
        colors = np.tile(
            np.array([0.1, 0.3, 1.0], dtype=np.float32), (*self.shape, self.max_num_vertices, 1)
        )
        for e in itertools.product(*self.shape_ranges):
            vertices[e][: n_vertices[e]] = np.asarray(self.meshes[e].vertices)[:, (0, 2, 1)] * self.scalings[e]
            faces[e][: n_faces[e]] = np.asarray(self.meshes[e].triangles)

        self.nV.from_numpy(n_vertices)
        self.nF.from_numpy(n_faces)
        self.V.from_numpy(vertices)
        self.F.from_numpy(faces)
        self.C.from_numpy(colors)

        # base object properties
        self.pos = ti.Vector.field(3, dtype=ti.f32, shape=self.shape)
        self.quat = ti.Vector.field(4, dtype=ti.f32, shape=self.shape)
        self.rot = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.shape)
        self.scale = ti.field(dtype=ti.f32, shape=self.shape)

        # rigid object properties
        self.type = ti.field(dtype=ti.i32, shape=self.shape)
        self.mass = ti.field(dtype=ti.f32, shape=self.shape)
        self.massInv = ti.field(dtype=ti.f32, shape=self.shape)
        self.inertia = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.shape)
        self.inertiaInv = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.shape)
        self.v = ti.Vector.field(3, dtype=ti.f32, shape=self.shape)  # linear velocity
        self.w = ti.Vector.field(3, dtype=ti.f32, shape=self.shape)  # angular velocity
        self.force = ti.Vector.field(3, dtype=ti.f32, shape=self.shape)  # force on body
        self.torque = ti.Vector.field(3, dtype=ti.f32, shape=self.shape)  # torque

        self.reset_members()

    @ti.kernel
    def reset_members(self):
        for idx in ti.grouped(self.V):
            self.new_V[idx] = self.V[idx]

        for idx in ti.grouped(self.pos):
            self.type[idx] = ObjType.DYNAMIC
            self.mass[idx] = 1.0
            self.massInv[idx] = 1.0 / self.mass[idx]
            self.pos[idx] = [0.0, 0.0, 0.0]
            self.quat[idx] = [0.0, 0.0, 0.0, 1.0]
            self.rot[idx] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            self.scale[idx] = 1.0
            self.v[idx] = [0.0, 0.0, 0.0]
            self.w[idx] = [0.0, 0.0, 0.0]
            self.force[idx] = [0.0, 0.0, 0.0]
            self.torque[idx] = [0.0, 0.0, 0.0]

    @ti.kernel
    def recompute_COM(self):
        com = ti.Vector([0.0, 0.0, 0.0])
        for idx in ti.grouped(self.pos):
            for i in range(self.nV[idx]):
                com += self.V[idx, i]
            com /= self.V.shape[-1]
            for i in range(self.nV[idx]):
                self.V[idx, i] -= com
            com = ti.Vector([0.0, 0.0, 0.0])

    @ti.func
    def get_type(self, idx=()):
        return self.type[idx]

    @ti.func
    def get_mass(self, idx=()):
        return self.mass[idx]

    @ti.func
    def get_massInv(self, idx=()):
        return self.massInv[idx]

    @ti.func
    def get_position(self, idx=()):
        return self.pos[idx]

    @ti.func
    def get_rotation(self, idx=()):
        return self.quat[idx]

    @ti.func
    def get_rotation_matrix(self, idx=()):
        return self.rot[idx]

    @ti.func
    def get_scale(self, idx=()):
        return self.scale[idx]

    @ti.func
    def get_inertia(self, idx=()):
        return self.inertia[idx]

    @ti.func
    def get_inertiaInv(self, idx=()):
        return self.inertiaInv[idx]

    @ti.func
    def get_inertia_world(self, idx=()):
        return self.rot[idx] @ self.inertia[idx] @ self.rot[idx].inverse()

    @ti.func
    def get_inertiaInv_world(self, idx=()):
        return self.rot[idx] @ self.inertiaInv[idx] @ self.rot[idx].inverse()

    @ti.func
    def get_linear_momentum(self, idx=()):
        return self.v[idx] * self.mass[idx]

    @ti.func
    def get_angular_momentum(self, idx=()):
        return self.get_inertia_world(idx) @ self.w[idx]

    @ti.func
    def get_linear_velocity(self, idx=()):
        return self.v[idx]

    @ti.func
    def get_angular_velocity(self, idx=()):
        return self.w[idx]

    @ti.func
    def get_velocity(self, p, idx=()):
        return self.get_linear_velocity(idx) + self.get_angular_velocity(idx).cross(
            p - self.pos[idx]
        )

    @ti.func
    def get_force(self, idx=()):
        return self.force[idx]

    @ti.func
    def get_torque(self, idx=()):
        return self.torque[idx]

    @ti.func
    def apply_force_to_COM(self, f, idx=()):
        if self.type[idx] == int(ObjType.DYNAMIC):
            self.force[idx] += f

    @ti.func
    def apply_force(self, f, p, idx=()):
        if self.type[idx] == int(ObjType.DYNAMIC):
            self.force[idx] += f
            self.torque[idx] += (p - self.pos[idx]).cross(f)

    @ti.func
    def apply_torque(self, t, idx=()):
        if self.type[idx] == int(ObjType.DYNAMIC):
            self.torque[idx] += t

    @ti.func
    def set_type(self, t, idx=()):
        self.type[idx] = t

        if self.type[idx] == ObjType.STATIC:
            self.mass[idx] = 3e38
            self.massInv[idx] = 0
            self.inertia[idx] = ti.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
            self.inertiaInv[idx] = ti.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
            self.force[idx] = ti.Vector([0.0, 0.0, 0.0])
            self.torque[idx] = ti.Vector([0.0, 0.0, 0.0])

    @ti.func
    def set_mass(self, m, idx=()):
        if self.type[idx] == int(ObjType.DYNAMIC):
            self.mass[idx] = m
            self.massInv[idx] = 1.0 / self.mass[idx]

    @ti.func
    def set_color(self, c, idx=()):
        for i in range(self.nV[idx]):
            self.C[(*idx, i)] = c

    @ti.func
    def set_colors(self, C, idx=()):
        for i in range(self.nV[idx]):
            self.C[(*idx, i)] = C[i]

    @ti.func
    def set_position(self, p, idx=()):
        self.pos[idx] = p

    @ti.func
    def set_rotation(self, q, idx=()):
        self.quat[idx] = q
        self.rot[idx] = utils.quaternion_to_matrix(q)

    @ti.func
    def set_rotation_matrix(self, R, idx=()):
        self.rot[idx] = R
        self.quat[idx] = utils.matrix_to_quaternion(R)

    @ti.func
    def set_scale(self, s, idx=()):
        self.scale[idx] = s

    @ti.func
    def set_inertia(self, _I, idx=()):
        if self.type[idx] == int(ObjType.DYNAMIC):
            self.inertia[idx] = _I
            self.inertiaInv[idx] = self.inertia[idx].inverse()

    @ti.func
    def set_linear_momentum(self, p, idx=()):
        if self.type[idx] == int(ObjType.DYNAMIC):
            self.v[idx] = self.massInv[idx] * p

    @ti.func
    def set_angular_momentum(self, _l, idx=()):
        if self.type[idx] == int(ObjType.DYNAMIC):
            self.w[idx] = self.get_inertiaInv_world(idx) @ _l

    @ti.func
    def set_linear_velocity(self, v, idx=()):
        if self.type[idx] == int(ObjType.DYNAMIC):
            self.v[idx] = v

    @ti.func
    def set_angular_velocity(self, w, idx=()):
        if self.type[idx] == int(ObjType.DYNAMIC):
            self.w[idx] = w

    @ti.func
    def set_force(self, f, idx=()):
        if self.type[idx] == int(ObjType.DYNAMIC):
            self.force[idx] = f

    @ti.func
    def set_torque(self, t, idx=()):
        if self.type[idx] == int(ObjType.DYNAMIC):
            self.torque[idx] = t

    @ti.func
    def reset_force(self):
        for idx in ti.grouped(self.force):
            self.force[idx] = ti.Vector([0.0, 0.0, 0.0])

    @ti.func
    def reset_torque(self):
        for idx in ti.grouped(self.torque):
            self.torque[idx] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def advance(self, dt: ti.f32):
        for idx in ti.grouped(self.pos):
            # advance linear movement
            self.set_linear_momentum(self.get_linear_momentum(idx) + dt * self.get_force(idx), idx)
            self.set_position(self.get_position(idx) + dt * self.get_linear_velocity(idx), idx)

            # advance rotation
            self.set_angular_momentum(self.get_angular_momentum(idx) + dt * self.get_torque(idx), idx)
            # quaternion-based angular velocity
            w_quat = utils.vector_to_quat(self.get_angular_velocity(idx))
            q = self.get_rotation(idx)
            q_new = q + 0.5 * dt * utils.quaternion_multiply(w_quat, q)
            self.set_rotation(q_new.normalized(), idx)

        # update vertex positions
        self.update_new_positions()
        self.reset_force()
        self.reset_torque()

    @ti.func
    def update_new_positions(self):
        for idx in ti.grouped(self.pos):
            for i in range(self.nV[idx]):
                self.new_V[idx, i] = (
                        self.rot[idx] @ (self.scale[idx] * self.V[idx, i]) + self.pos[idx]
                )

    def update_meshes(self):
        # One-way binding from taichi vertices to o3d meshes
        new_V = self.new_V.to_numpy()
        C = self.C.to_numpy()
        for e in itertools.product(*self.shape_ranges):
            self.meshes[e].vertices = o3d.utility.Vector3dVector(
                new_V[e][: self.nV[e], (0, 2, 1)] * screen_to_world_ratio
            )
            self.meshes[e].vertex_colors = o3d.utility.Vector3dVector(
                C[e][: self.nV[e]]
            )


@ti.data_oriented
class Cube(RigidObjectField):
    """
    A RigidObjectField with only some simple cubes
    """

    def __init__(self):
        super().__init__(['./data/cube.off', './data/rect.off'])
        # set initial condition of simulation
        self.cur_step = 0
        self.t = 0.0
        self.dt = time_delta
        self._set_sim_init()
        self.update_meshes()

    @ti.kernel
    def _set_sim_init(self):
        for idx in ti.grouped(self.pos):
            self.set_inertia(
                self.get_mass(idx) * 2.0 / 6.0
                * ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                idx)  # inertia of cube
        self.set_position(ti.Vector([0.2 * boundary[0], 0.5 * boundary[1], 0.5 * boundary[2]]), (0))
        self.set_position(ti.Vector([0.6 * boundary[0], 0.5 * boundary[1], 0.5 * boundary[2]]), (1))
        self.update_new_positions()

    def reinitialize(self):
        # reset non taichi-scope variables here
        self.t = 0.0
        self.cur_step = 0
        self.reset_members()
        self._set_sim_init()
        self.update_meshes()

    @ti.kernel
    def apply_gravity(self):
        for idx in ti.grouped(self.mass):
            self.apply_force(ti.Vector([0.0, 0.0, -g_const]) * self.get_mass(idx), self.get_position(idx), idx)

    def step(self):
        self.apply_gravity()
        self.advance(self.dt)
        self.update_meshes()
        self.t += self.dt
        self.cur_step += 1


@ti.data_oriented
class Ball(RigidObjectField):
    """
    A RigidObjectField with only balls
    """

    def __init__(self):
        N = 30
        super().__init__(['./data/sphere.off'] * N, 1 + 3 * np.random.random((N,)))
        # set initial condition of simulation
        self.eps = rigid_rigid_eps
        self.cur_step = 0
        self.t = 0.0
        self.dt = time_delta
        self.radius = ti.field(float, self.shape)
        print('radius:', self.scalings)
        self.radius.from_numpy(self.scalings)
        self.has_collision_update = ti.field(int, self.shape)
        self.num_collisions = ti.field(int, ())
        self.energy = ti.field(float, ())
        self.p_after_collision = ti.Vector.field(3, float, self.shape)
        self.v_after_collision = ti.Vector.field(3, float, self.shape)
        self._set_sim_init()
        self.update_meshes()

    @ti.kernel
    def _set_sim_init(self):
        # for i in range(3):
        for I in ti.grouped(self.mass):
            self.set_mass(100 * self.radius[I], I)
            self.set_inertia(utils.inertia_ball(self.get_mass(I), self.radius[I]), I)  # inertia of ball
            self.set_position(ti.Vector([
                (0.2 + 0.6 * ti.random()) * boundary[0],
                (0.2 + 0.6 * ti.random()) * boundary[1],
                (0.4 + 0.2 * ti.random()) * boundary[2]
            ]), I)
            v = 20.0 * ti.random()
            phi = 2 * np.pi * ti.random()
            self.set_linear_velocity(ti.Vector([v * ti.cos(phi), v * ti.sin(phi), 0]), I)

        self.update_new_positions()

    def reinitialize(self):
        # reset non taichi-scope variables here
        self.t = 0.0
        self.cur_step = 0
        self.reset_members()
        self._set_sim_init()
        self.update_meshes()

    @ti.kernel
    def apply_gravity(self):
        for I in ti.grouped(self.mass):
            self.apply_force(ti.Vector([0.0, 0.0, -g_const]) * self.get_mass(I), self.get_position(I), I)

    @ti.kernel
    def detect_collision(self, eps: ti.f32):
        self.num_collisions[None] = 0
        for I in ti.grouped(self.has_collision_update):
            self.has_collision_update[I] = 0

        # for i, j in ti.ndrange(*(self.shape * 2)):
        #     if j == i:
        #         continue
        #     I = (i,)
        #     J = (j,)
        for I in ti.grouped(ti.ndrange(*self.shape)):
            for J in ti.grouped(ti.ndrange(*self.shape)):
                if all(I == J): continue
                mi = self.get_mass(I)
                mj = self.get_mass(J)
                ci = self.get_position(I)
                cj = self.get_position(J)
                ri = self.radius[I]
                rj = self.radius[J]
                cij = ci - cj
                dis = cij.norm()
                if dis < ri + rj:
                    # Perform sphere-sphere collision
                    self.num_collisions[None] += 1
                    self.has_collision_update[I] = 1
                    # self.has_collision_update[J] = 1
                    normal = cij / dis  # points from j to i
                    vi = self.get_linear_velocity(I)
                    vj = self.get_linear_velocity(J)
                    vi, vj = utils.sphere_collide_sphere(mi, mj, vi, vj, normal, eps)
                    self.v_after_collision[I] = vi
                    # self.v_after_collision[J] = vj
                    intersection = ri + rj - dis + epsilon
                    self.p_after_collision[I] = ci + intersection / 2 * normal
                    # self.p_after_collision[J] = cj - intersection / 2 * normal

        for I in ti.grouped(self.pos):
            if self.has_collision_update[I]:
                self.set_position(self.p_after_collision[I], I)
                self.set_linear_velocity(self.v_after_collision[I], I)

    @ti.kernel
    def compute_energy(self):
        self.energy[None] = 0
        for I in ti.grouped(self.mass):
            self.energy[None] += 0.5 * self.mass[I] * self.v[I].norm_sqr() + self.mass[I] * g_const * self.pos[I].z

    def step(self):
        self.apply_gravity()
        self.detect_collision(self.eps)
        self.advance(self.dt)
        self.compute_energy()
        self.update_meshes()
        self.t += self.dt
        self.cur_step += 1
        collisions = self.num_collisions.to_numpy()
        if collisions > 0:
            print('# collisions:', collisions, '  energy:', self.energy.to_numpy())

    @ti.func
    def get_AABB(self, idx):
        """
        Return the AABB represented by a shape (6,) vector - first three entries indicate bmin.xyz, last three bmax.xyz
        The provided AABB is converted to grid indices with boundary checking
        """
        p = self.pos[idx]
        r = self.radius[idx]

        aabb = ti.Vector([p.x - r, p.y - r, p.z - r,
                          p.x + r, p.y + r, p.z + r])
        ret = ti.Vector([0, 0, 0, 0, 0, 0], ti.i32)

        for i in ti.static(range(3)):
            ret[i] = utils.clamp(utils.get_cell(aabb[i]), 0, grid_size[i])
            ret[3 + i] = utils.clamp(utils.get_cell(aabb[3 + i]) + 1, 0, grid_size[i])

        return ret
