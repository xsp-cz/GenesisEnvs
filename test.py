import pyrender
import trimesh
import numpy as np
import os
import matplotlib.pyplot as plt

os.environ["PYOPENGL_PLATFORM"] = "osmesa"

# # Create a simple mesh, for example, a sphere
# mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)

# # Convert the mesh into a pyrender Mesh object
# mesh = pyrender.Mesh.from_trimesh(mesh)

# # Create a scene and add the mesh to it
# scene = pyrender.Scene()
# scene.add(mesh)

mesh = trimesh.load('fuse.obj')
mesh = pyrender.Mesh.from_trimesh(mesh)
scene = pyrender.Scene()
scene.add(mesh)

camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
s = np.sqrt(2)/2
camera_pose = np.array([
    [0.0, -s,   s,   0.3],
    [1.0,  0.0, 0.0, 0.0],
    [0.0,  s,   s,   0.35],
    [0.0,  0.0, 0.0, 1.0],
])
scene.add(camera, pose=camera_pose)
light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                            innerConeAngle=np.pi/16.0,
                            outerConeAngle=np.pi/6.0)
scene.add(light, pose=camera_pose)
r = pyrender.OffscreenRenderer(1600, 1200)
color, depth = r.render(scene)

plt.figure()
plt.subplot(1,2,1)
plt.axis('off')
plt.imshow(color)
plt.subplot(1,2,2)
plt.axis('off')
#plt.imshow(depth, cmap=plt.cm.gray_r)
plt.imshow(depth, 'rainbow')
plt.savefig('test.jpg')

# # Setup camera parameters
# camera = pyrender.PerspectiveCamera(yfov=np.pi)

# # Define the camera pose (translation and rotation)
# camera_pose = np.eye(4)
# camera_pose[:3, 3] = np.array([3, 3, 3])  # Camera is placed at (3, 3, 3)
# scene.add(camera, pose=camera_pose)

# # # Add a directional light
# # light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
# # scene.add(light, pose=camera_pose)

# # Create a simple renderer using OSMesa as the backend
# # Ensure OSMesa is installed and available in your environment
# renderer = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)

# # Render the scene offscreen
# color, depth = renderer.render(scene)

# # Save the rendered image to a file
# import imageio
# imageio.imwrite('rendered_image.bmp', color)
