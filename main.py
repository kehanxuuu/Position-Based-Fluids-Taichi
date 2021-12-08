import time

import open3d as o3d

from pbf import *

from matplotlib import cm


def render(fluid: Fluid, vis, pcd, box):
    pos_np = fluid.positions.to_numpy()
    pos_np *= screen_to_world_ratio
    pos_np = pos_np[:, (0, 2, 1)]  # recap: z and y axis in the simulation are swapped for better visualization
    pcd.points = o3d.utility.Vector3dVector(pos_np)
    if particle_color == 'velocity':
        velnorm_np = np.linalg.norm(fluid.velocities.to_numpy(), axis=1) / cm_max_velocity
        pcd.colors = o3d.utility.Vector3dVector(cm.jet(velnorm_np)[:, :3])
    elif particle_color == 'density':
        fluid.find_neighbour()
        fluid.compute_density()
        density_np = fluid.density.to_numpy()
        print(f'{density_np.min()=}, {density_np.max()=}, {density_np.mean()=}')
        # if (density_np < rho0).any():
        #     print('Exists density smaller than rho0')
        density_np = density_np / rho0 * 0.5  # map to [0, 1]
        pcd.colors = o3d.utility.Vector3dVector(cm.RdBu(density_np)[:, :3])
    elif particle_color == 'vorticity':
        omegas_np = np.linalg.norm(fluid.omegas.to_numpy(), axis=1) / cm_max_vorticity
        pcd.colors = o3d.utility.Vector3dVector(cm.YlGnBu(omegas_np)[:, :3])
    else:
        raise ValueError('unknown particle color key')
    vis.update_geometry(pcd)
    box.translate(np.array([fluid.board_states[None][0], boundary[1] / 2, boundary[2] / 2]) * screen_to_world_ratio, relative=False)
    vis.update_geometry(box)


def main():
    ti.init(arch=ti.gpu)
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

    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    box = o3d.geometry.TriangleMesh.create_box(5, screen_res[1], screen_res[2])
    box.translate(np.array([screen_res[0], 0, 0]))
    vis.add_geometry(box)

    fluid = Fluid()
    fluid.init_particles()
    print(f'boundary={boundary} grid={grid_size} cell_size={cell_size}')

    iter = 0
    while True:
        if control.reset:
            fluid.init_particles()
            resetSimulation(vis)
            print(f'reset simulation')

        if not control.paused:
            start_time = time.time()
            fluid.move_board()
            fluid.run_pbf()
            render(fluid, vis, pcd, box)
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
