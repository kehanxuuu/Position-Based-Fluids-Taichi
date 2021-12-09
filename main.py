import time

from pbf import *
from rigidbody import Cube


def render(vis, fluid: Fluid, cube: Cube, box):
    vis.update_geometry(fluid.pcd)
    box.translate(np.array([fluid.board_states[None][0], boundary[1] / 2, boundary[2] / 2]) * screen_to_world_ratio, relative=False)
    vis.update_geometry(box)
    for mesh in cube.meshes.ravel():
        vis.update_geometry(mesh)


def main():
    ti.init(arch=ti.gpu if device == 'gpu' else ti.cpu)

    # setup gui
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    class GUIState:
        def __init__(self):
            self.reset = False
            self.paused = False

    control = GUIState()

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
        global xsph_c
        xsph_c += 0.05

    def decreaseViscosity(_):
        global xsph_c
        xsph_c -= 0.05
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

    box = o3d.geometry.TriangleMesh.create_box(5, screen_res[1], screen_res[2])
    box.translate(np.array([screen_res[0], 0, 0]))
    vis.add_geometry(box)

    fluid = Fluid()
    fluid.init_particles()
    fluid.update_point_cloud()
    cube = Cube()

    vis.add_geometry(fluid.pcd)
    for mesh in cube.meshes.ravel():
        print(mesh)
        vis.add_geometry(mesh)

    print(f'boundary={boundary} grid={grid_size} cell_size={cell_size}')

    iter = 0
    while True:
        if control.reset:
            fluid.init_particles()
            cube.reinitialize()
            resetSimulation(vis)
            print(f'reset simulation')

        if not control.paused:
            start_time = time.time()

            fluid.move_board()
            fluid.run_pbf()
            fluid.update_point_cloud()
            cube.step()

            render(vis, fluid, cube, box)
            if iter % 20 == 1:
                time_interval = time.time() - start_time
                fluid.print_stats(time_interval)
            iter += 1

        if not vis.poll_events():
            break

        if not control.paused:
            vis.update_renderer()


if __name__ == '__main__':
    main()