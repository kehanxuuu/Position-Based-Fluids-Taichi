# ./Blender -b -P 'C:\Users\Kehan Xu\Downloads/blender_render_video_sequence.py'
import bpy
from mathutils import Vector
import math
import os
import time
import logging
import json
 
def euler_from_quaternion(input):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    x = input[0]
    y = input[1]
    z = input[2]
    w = input[3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    return roll_x, pitch_y, yaw_z # in radians

def listdir_nohidden(path):
    visible = []
    for f in os.listdir(path):
        if not f.startswith('.'):
            visible.append(f)
    return visible

logging.basicConfig(level=logging.DEBUG,
                    filename='C:/Users/Kehan Xu/Downloads/render_time.log',
                    filemode='a', # a: add on, w: cover
                    format= '%(message)s'
                    # '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )

dir_path_mesh = "C:/Users/Kehan Xu/Documents/ETH/HS2021/Physically Based Simulation/fluid-fans/meshes/"
dir_path_rigid_json = "C:/Users/Kehan Xu/Documents/ETH/HS2021/Physically Based Simulation/fluid-fans/rigids/"
dir_path_img = "C:/Users/Kehan Xu/Documents/ETH/HS2021/Physically Based Simulation/fluid-fans/rendered_imgs/"
for filename in os.listdir(dir_path_img):
    os.remove(dir_path_img+filename)

file_path_video = "C:/Users/Kehan Xu/Documents/ETH/HS2021/Physically Based Simulation/fluid-fans/rendered_imgs/water_video.avi"
# obj_name = "frame_110_surface"
obj_name_list = sorted(listdir_nohidden(dir_path_mesh))
obj_list_num = len(obj_name_list)
# obj_path = dir_path+obj_name+".obj"
obj_path_list = [dir_path_mesh+obj_name for obj_name in obj_name_list]
# render_img_path = dir_path+obj_name+"_render_total.png"
json_path_list = []
render_img_path = []
obj_name_list_wo_ext = []
for obj_name in obj_name_list:
    fileName, fileExtension = os.path.splitext(obj_name)
    obj_name_list_wo_ext.append(fileName)
    render_img_path.append(dir_path_img+fileName+"_render.png")
    json_path_list.append(dir_path_rigid_json+fileName+".json")

sphere_obj_path = "C:/Users/Kehan Xu/Documents/ETH/HS2021/Physically Based Simulation/fluid-fans/data/sphere.obj"
torus_obj_path = "C:/Users/Kehan Xu/Documents/ETH/HS2021/Physically Based Simulation/fluid-fans/data/torus.obj"
sphere_tex_path = "C:/Users/Kehan Xu/Downloads/Sphere.jpg" # sphere_tex_tmp
sphere_tex_path_2 = "C:/Users/Kehan Xu/Downloads/Sphere_2.jpg"
torus_tex_path = "C:/Users/Kehan Xu/Downloads/Torus.jpg"
plane_tex_path = "C:/Users/Kehan Xu/Downloads/ceramic_tile.jpg"
environment_tex_path = "C:/Users/Kehan Xu/Downloads/Skies-001.jpg"

scale_ratio = [0.091, 0.091, 0.091]

# command_line = False

def look_at(obj_camera, point=Vector((0, 0, 0))):
    loc_camera = obj_camera.location
    direction = point - loc_camera
    rot_quat = direction.to_track_quat('-Z', 'Y')
    obj_camera.rotation_euler = rot_quat.to_euler()


scene = bpy.context.scene
light_data = bpy.data.objects['Light']
light_data.select_set(True)
bpy.ops.object.delete()
try:
    cube = bpy.data.objects['Cube']
    cube.select_set(True)
    bpy.ops.object.delete()
except:
    pass


camera = bpy.data.objects['Camera']
camera.location = (16.68, -14.07, 10.20)
camera.rotation_euler = (math.radians(62), math.radians(-2.91), math.radians(45.6))
# look_at(camera, Vector((0, 0, 0)))
# camera.select_set(True)
# if command_line:
#     ov=bpy.context.copy()
#     ov['area']=[a for a in bpy.context.screen.areas if a.type=="VIEW_3D"][0]
#     bpy.ops.transform.rotate(ov, value=math.radians(3), orient_axis='Z')
#     bpy.ops.transform.rotate(ov, value=math.radians(1.1), orient_axis='Y')
# else:
#     bpy.ops.transform.rotate(value=math.radians(3), orient_axis='Z')
#     bpy.ops.transform.rotate(value=math.radians(1.1), orient_axis='Y')


# create water material
mat = bpy.data.materials.new(name="WaterMaterial")
mat.use_nodes = True
mat.node_tree.nodes.new(type='ShaderNodeBsdfGlass')
mat.node_tree.nodes['Glass BSDF'].inputs['Roughness'].default_value = 0.0
mat.node_tree.nodes['Glass BSDF'].inputs['IOR'].default_value = 1.33
inp = mat.node_tree.nodes['Material Output'].inputs['Surface']
outp = mat.node_tree.nodes['Glass BSDF'].outputs['BSDF']
mat.node_tree.links.new(inp,outp)


# create sphere material
mat = bpy.data.materials.new(name="SphereMaterial")
mat.use_nodes = True
mat.node_tree.nodes.new(type='ShaderNodeBsdfDiffuse')
# mat.node_tree.nodes['Diffuse BSDF'].inputs['Color'].default_value = (1, 0, 0.07, 1)
node_tex = mat.node_tree.nodes.new(type='ShaderNodeTexImage')
node_tex.image = bpy.data.images.load(sphere_tex_path)
inp = mat.node_tree.nodes['Material Output'].inputs['Surface']
outp = mat.node_tree.nodes['Diffuse BSDF'].outputs['BSDF']
mat.node_tree.links.new(inp,outp)
inp = mat.node_tree.nodes['Diffuse BSDF'].inputs['Color']
outp = node_tex.outputs['Color']
mat.node_tree.links.new(inp,outp)

#ceate sphere material 2 by copying sphere & change tex
mat = bpy.data.materials["SphereMaterial"].copy()
mat.name = "SphereMaterial2"
node_tex = mat.node_tree.nodes['Image Texture']
node_tex.image = bpy.data.images.load(sphere_tex_path_2)


#ceate torus material by copying sphere & change tex
mat = bpy.data.materials["SphereMaterial"].copy()
mat.name = "TorusMaterial"
node_tex = mat.node_tree.nodes['Image Texture']
node_tex.image = bpy.data.images.load(torus_tex_path)

# create plane material
mat = bpy.data.materials.new(name="PlaneMaterial")
mat.use_nodes = True
mat.node_tree.nodes.new(type='ShaderNodeBsdfGlossy')
# mat.node_tree.nodes['Diffuse BSDF'].inputs['Color'].default_value = (1, 0, 0.07, 1)
mat.node_tree.nodes['Glossy BSDF'].inputs['Roughness'].default_value = 0.1
node_tex = mat.node_tree.nodes.new(type='ShaderNodeTexImage')
node_tex.image = bpy.data.images.load(plane_tex_path)
inp = mat.node_tree.nodes['Material Output'].inputs['Surface']
outp = mat.node_tree.nodes['Glossy BSDF'].outputs['BSDF']
mat.node_tree.links.new(inp,outp)
inp = mat.node_tree.nodes['Glossy BSDF'].inputs['Color']
outp = node_tex.outputs['Color']
mat.node_tree.links.new(inp,outp)
# set plane
bpy.ops.mesh.primitive_plane_add(size=51.7, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(0.103, 0.103, 0.103))
plane = bpy.data.objects['Plane']
plane.data.materials.clear()
plane.data.materials.append(bpy.data.materials["PlaneMaterial"])


# set background image
scene.world.use_nodes = True
tree_nodes = scene.world.node_tree.nodes
tree_nodes.clear()
node_background = tree_nodes.new(type='ShaderNodeBackground')
node_environment = tree_nodes.new('ShaderNodeTexEnvironment')
node_environment.image = bpy.data.images.load(environment_tex_path)
# node_environment.location = -300,0
node_output = tree_nodes.new(type='ShaderNodeOutputWorld')   
# node_output.location = 200,0
# Link all nodes
links = scene.world.node_tree.links
link = links.new(node_environment.outputs["Color"], node_background.inputs["Color"])
link = links.new(node_background.outputs["Background"], node_output.inputs["Surface"])


# set light
bpy.ops.object.light_add(type='SUN')
light_ob = bpy.context.object
light = light_ob.data
light.energy = 1
light_ob.location = (3.644, 15.456, 12.611)
light_ob.rotation_euler = (math.radians(40.5), math.radians(46), math.radians(143))
# light_ob.select_set(True)
# if command_line:
#     ov=bpy.context.copy()
#     ov['area']=[a for a in bpy.context.screen.areas if a.type=="VIEW_3D"][0]
#     bpy.ops.transform.rotate(ov, value=math.radians(-40.5), orient_axis='X')
#     bpy.ops.transform.rotate(ov, value=math.radians(-46), orient_axis='Y')
#     bpy.ops.transform.rotate(ov, value=math.radians(-143), orient_axis='Z')
# else:
#     bpy.ops.transform.rotate(value=math.radians(-40.5), orient_axis='X')
#     bpy.ops.transform.rotate(value=math.radians(-46), orient_axis='Y')
#     bpy.ops.transform.rotate(value=math.radians(-143), orient_axis='Z')


# render settings
bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "OPTIX"
scene.cycles.device = 'GPU'
scene.render.engine = 'CYCLES' # 'BLENDER_EEVEE'
scene.render.film_transparent = True
# (1920 1080) (1280 720) (960 540) (640 360)
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080
scene.render.image_settings.file_format = 'PNG'

logging.debug("Render with resolutin {} {}.".format(scene.render.resolution_x, scene.render.resolution_y))
frame_start_time = time.time()
total_start_time = time.time()
spheres = []
toruses = []
for i in range(1):
# for i in range(obj_list_num):
    # load dict
    with open(json_path_list[i],'r') as load_f:
        rigid_dict = json.load(load_f)
    if i == 0:
        # load spheres and toruses
        n_balls = rigid_dict['n_balls']
        n_toruses = rigid_dict['n_toruses']
        scalings = rigid_dict['scalings']
        for j in range(n_balls):
            bpy.ops.import_scene.obj(filepath=sphere_obj_path)
            name = ""
            if j != 0:
                name = ".00{}".format(j) # should not exceed 10
            spheres.append(bpy.data.objects['Sphere'+name])
            spheres[j].select_set(True)
            bpy.ops.transform.resize(value=(scale_ratio[0]*scalings[j], scale_ratio[1]*scalings[j], scale_ratio[2]*scalings[j]))
            spheres[j].data.materials.clear()
            sphereMat = "SphereMaterial" if j % 2 == 0 else "SphereMaterial2"
            spheres[j].data.materials.append(bpy.data.materials[sphereMat])
        for j in range(n_toruses):
            bpy.ops.import_scene.obj(filepath=torus_obj_path)
            name = ""
            if j != 0:
                name = ".00{}".format(j) # should not exceed 10
            toruses.append(bpy.data.objects['Torus'+name])
            toruses[j].select_set(True)
            bpy.ops.transform.resize(value=(scale_ratio[0]*scalings[j+n_balls], scale_ratio[1]*scalings[j+n_balls], scale_ratio[2]*scalings[j+n_balls]))
            toruses[j].data.materials.clear()
            toruses[j].data.materials.append(bpy.data.materials["TorusMaterial"])
    # set sphere and torus positions and rotations
    rigid_pos = rigid_dict['pos']
    rigid_quat = rigid_dict['quat']
    for j in range(n_balls):
        spheres[j].location = (rigid_pos[j][0]*scale_ratio[0], -rigid_pos[j][1]*scale_ratio[1], rigid_pos[j][2]*scale_ratio[2])
        angle_x, angle_y, angle_z = euler_from_quaternion(rigid_quat[j]) # in radians
        spheres[j].rotation_euler = (angle_x, angle_y, angle_z)
    for j in range(n_toruses):
        toruses[j].location = (rigid_pos[j+n_balls][0]*scale_ratio[0], -rigid_pos[j+n_balls][1]*scale_ratio[1], rigid_pos[j+n_balls][2]*scale_ratio[2])
        angle_x, angle_y, angle_z = euler_from_quaternion(rigid_quat[j+n_balls]) # in radians
        toruses[j].rotation_euler = (angle_x, angle_y, angle_z)
    # load water
    obj_name = obj_name_list_wo_ext[i]
    bpy.ops.import_scene.obj(filepath=obj_path_list[i]) # import water
    water = bpy.data.objects[obj_name]
    # set water info
    water.select_set(True)
    bpy.ops.transform.resize(value=(scale_ratio[0], scale_ratio[1], scale_ratio[2]))
    # bpy.ops.transform.rotate(value=3.14, orient_axis='Z')
    # bpy.ops.transform.translate(value=(0, 0, 0))
    # add material to water
    water.data.materials.clear()
    water.data.materials.append(bpy.data.materials["WaterMaterial"])       
    # render
    scene.render.filepath = render_img_path[i]
    bpy.ops.render.render(write_still=True, use_viewport=False)
    water.select_set(True)
    # bpy.ops.object.delete()
    delta = time.time() - frame_start_time
    frame_start_time = time.time()
    logging.debug("Finished rendering frame {} / {}, used {} seconds".format(i, obj_list_num, delta))

logging.debug("Finished rendering total {} frames in {} seconds.".format(obj_list_num, time.time() - total_start_time))
logging.debug("    ") # empty line to separate the next log

scene.render.use_sequencer = True
scene.sequence_editor_create()

for i in range (obj_list_num):
    scene.sequence_editor.sequences.new_image(
		name=obj_name_list_wo_ext[i],
		filepath=render_img_path[i],
		channel=1, frame_start=i)

scene.frame_end = obj_list_num
scene.render.image_settings.file_format = 'AVI_JPEG' 
scene.render.filepath = file_path_video
bpy.ops.render.render( animation=True )