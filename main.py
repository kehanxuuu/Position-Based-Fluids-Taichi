import time
import os

from pbf import *
from rigidbody import *
import utils_normal


def render(vis, fluid: Fluid, rigid: RigidObjectField, box, box2):
    vis.update_geometry(fluid.pcd)
    box.translate(np.array([fluid.board_states[0], boundary[2] / 2, boundary[1] / 2]) * screen_to_world_ratio, relative=False)
    vis.update_geometry(box)
    box2.translate(np.array([fluid.board_states[4], boundary[2] / 2, boundary[1] / 2]) * screen_to_world_ratio, relative=False)
    vis.update_geometry(box2)
    for mesh in rigid.meshes.ravel():
        vis.update_geometry(mesh)


def main():
    ti.init(arch=ti.gpu if device == 'gpu' else ti.cpu, random_seed=0)

    # setup gui
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    class GUIState:
        def __init__(self):
            self.reset = False
            self.paused = False

    control = GUIState()

    rigid = SimpleGeometryRigid(5, 10, 1.5, 1.8)
    for mesh in rigid.meshes.ravel():
        vis.add_geometry(mesh)

    fluid = Fluid(rigid)
    fluid.init_particles()
    fluid.update_point_cloud()
    vis.add_geometry(fluid.pcd)

    def resetSimulation(_):
        control.reset ^= 1

    def pause(_):
        control.paused ^= 1

    def increaseVorticity(_):
        global vorticity_epsilon
        vorticity_epsilon += 0.05

    def decreaseVorticity(_):
        global vorticity_epsilon
        vorticity_epsilon -= 0.05
        if vorticity_epsilon <= 0:
            vorticity_epsilon = 0

    def increaseViscosity(_):
        fluid.xsph_c += 0.05

    def decreaseViscosity(_):
        fluid.xsph_c -= 0.05
        fluid.xsph_c = max(0, fluid.xsph_c)

    def increaseCollisionEps(_):
        fluid.rigid_boundary_eps += 0.1
        rigid.eps += 0.1

    def decreaseCollisionEps(_):
        fluid.rigid_boundary_eps = max(fluid.rigid_boundary_eps - 0.1, 0)
        rigid.eps = max(rigid.eps - 0.1, 0)

    def increaseStiffness(_):
        fluid.rigid_boundary_stiffness *= 2

    def decreaseStiffness(_):
        fluid.rigid_boundary_stiffness /= 2

    def increaseBoardOmega(_):
        fluid.adjust_board_omega(1.1)

    def decreaseBoardOmega(_):
        fluid.adjust_board_omega(1.0 / 1.1)

    def make_particle_color_callback(color: str):
        def callback(_):
            fluid.particle_color = color

        return callback

    vis.register_key_callback(ord("R"), resetSimulation)
    vis.register_key_callback(ord(" "), pause)  # space
    vis.register_key_callback(ord("E"), increaseVorticity)
    vis.register_key_callback(ord("D"), decreaseVorticity)
    vis.register_key_callback(ord("W"), increaseViscosity)
    vis.register_key_callback(ord("S"), decreaseViscosity)
    vis.register_key_callback(ord("2"), increaseCollisionEps)
    vis.register_key_callback(ord("1"), decreaseCollisionEps)
    vis.register_key_callback(ord("4"), increaseBoardOmega)
    vis.register_key_callback(ord("3"), decreaseBoardOmega)
    vis.register_key_callback(ord("5"), make_particle_color_callback('velocity'))
    vis.register_key_callback(ord("6"), make_particle_color_callback('density'))
    vis.register_key_callback(ord("7"), make_particle_color_callback('vorticity'))

    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=100, origin=[0, 0, 0]
    )
    vis.add_geometry(coordinate_frame)  # coordinate frame

    aabb = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=np.array([0, 0, 0]), max_bound=np.array(screen_res)
    )
    aabb.color = [0.7, 0.7, 0.7]
    vis.add_geometry(aabb)  # bounding box

    box = o3d.geometry.TriangleMesh.create_box(3, screen_res[1], screen_res[2])
    box.translate(np.array([screen_res[0], screen_res[1] / 2, screen_res[2] / 2]), relative=False)
    vis.add_geometry(box)

    box2 = o3d.geometry.TriangleMesh.create_box(3, screen_res[1], screen_res[2])
    box2.translate(np.array([0, screen_res[1] / 2, screen_res[2] / 2]), relative=False)
    vis.add_geometry(box2)

    print(f'boundary={boundary} grid={grid_size} cell_size={cell_size}')

    iter = 0
    if output_mesh:
        particle_dir = "particles"
        rigid_dir = "rigids"
        mesh_dir = "meshes"
        frame_start = 150
        frame_end = 450

        os.makedirs(particle_dir, exist_ok=True)
        os.makedirs(mesh_dir, exist_ok=True)

        if clear_particle_mesh_directory:
            utils_normal.clear_directory(particle_dir)
            utils_normal.clear_directory(rigid_dir)
            utils_normal.clear_directory(mesh_dir)

    while True:
        if control.reset:
            fluid.init_particles()
            rigid.reinitialize()
            resetSimulation(vis)
            print(f'reset simulation')

        if not control.paused:
            start_time = time.time()

            fluid.move_board()
            fluid.run_pbf()
            rigid.step()

            if iter % 1 == 0:
                time_interval = time.time() - start_time
                fluid.print_stats(iter, time_interval)
                fluid.update_point_cloud()
                rigid.update_meshes()
                render(vis, fluid, rigid, box, box2)
                vis.update_renderer()

            if output_mesh:
                if iter >= frame_start and iter < frame_end:
                    filename = "frame_{:0>5d}".format(iter)
                    filepath_particle = os.path.join(particle_dir, filename)
                    filepath_rigid = os.path.join(rigid_dir, filename)
                    pos_np = fluid.positions.to_numpy()
                    pos_np = pos_np[:, (0, 2, 1)]
                    utils_normal.convert_particle_info_to_json(pos_np, filepath_particle)
                    utils_normal.convert_json_to_mesh_command_line(particle_dir, mesh_dir, filename)

                    rigid_pos_np = rigid.pos.to_numpy() # without [:, (0, 2, 1)] here!
                    rigid_quat_np = rigid.quat.to_numpy() # quat
                    rigid_dict_json = {
                        "n_balls": rigid.n_balls,
                        "n_toruses": rigid.n_toruses,
                        "scalings": rigid.scalings.tolist(),
                        "pos": rigid_pos_np.tolist(),
                        "quat": rigid_quat_np.tolist()
                    }
                    utils_normal.convert_rigid_info_to_json(rigid_dict_json, filepath_rigid)
                    

            iter += 1

        if not vis.poll_events():
            break


if __name__ == '__main__':
    main()
