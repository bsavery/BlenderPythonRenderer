import taichi as ti
from . import camera
from . import integrator
from . import instance
from . import mesh
from . import material
from .vector import *
from .ray import Ray
import numpy as np


InFlightRay = ti.types.struct(depth=ti.i32, ray=Ray, throughput=Vector4)


# Taichi data node
DATA = None


# render constants
NUM_SAMPLES = 64
MAX_DEPTH = 8
WIDTH = 512
HEIGHT = 512
_inited = False


def setup_render(exported_scene, width, height, samples, max_depth):
    print('call setup')
    global _inited
    if not _inited:
        ti.init(arch=ti.gpu)
        _inited = True

    ''' Creates taichi data fields from numpy arrays exported from Blender '''
    camera.setup_data(exported_scene.camera)
    mesh.setup_data(exported_scene.meshes)
    material.setup_data(exported_scene.materials)
    instance.setup_data(exported_scene.instances)

    # setup the pixel buffer and inflight rays
    global DATA, pixel_buffer, sample_count, rays_in_flight
    
    if DATA is None:
        # framebuffer RGBA
        pixel_buffer = ti.Vector.field(n=4, dtype=ti.f32)
        # sample counts for each pixel
        sample_count = ti.field(dtype=ti.i32)
        # rays in flight
        rays_in_flight = InFlightRay.field()

        DATA = ti.root.dense(ti.ij, (width, height))
        DATA.place(pixel_buffer, sample_count, rays_in_flight)

    pixel_buffer.from_numpy(np.zeros((width, height, 4), dtype=np.float32))
    sample_count.from_numpy(np.zeros((width, height), dtype=np.int32))
    rays_in_flight.depth.from_numpy(np.zeros((width, height), dtype=np.int32))

    # constants
    global NUM_SAMPLES, MAX_DEPTH, WIDTH, HEIGHT
    NUM_SAMPLES = samples
    MAX_DEPTH = max_depth
    WIDTH = width
    HEIGHT = height


@ti.kernel
def render_pass() -> ti.i32:
    ''' render one ray bounce for every pixel in the image
        for each pixel in the image we
            1.  Check if it needs a new camera ray
            2.  Trace one more bounce
            3.  check if need to stop, accumulate, etc

        RETURNS num samples completed
    '''
    samples_done = 0
    for i, j in pixel_buffer:
        if sample_count[i, j] == NUM_SAMPLES:
            # skip this pixel
            continue

        inflight = rays_in_flight[i, j]
        ray = inflight.ray
        depth = inflight.depth
        throughput = inflight.throughput

        if depth == 0:
            # add a new ray for pixel that needs new one
            s = (i + ti.random()) / (WIDTH - 1)
            t = (j + ti.random()) / (HEIGHT - 1)
            ray = camera.get_ray(s, t)
            depth = MAX_DEPTH
            throughput = Vector4(1.0)

        # integrate one bounce along path
        ray, throughput, ray_stop = integrator.trace_ray(ray, throughput, Vector4(0.0))
        depth -= 1

        # reset inflight for this pixel if stopped
        if ray_stop or depth == 0:
            # somewhat hack to set alpha to 1 if more than one bounce in
            if MAX_DEPTH - inflight.depth > 0:
                throughput.w = 1.0

            # add accumulated color to pixel
            pixel_buffer[i, j] += throughput
            sample_count[i, j] += 1
            samples_done += 1
            depth = 0

        rays_in_flight[i, j] = InFlightRay(depth=depth, ray=ray, throughput=throughput)

    return samples_done


def get_buffer():
    # get the framebuffer and divide by sample count
    buffer = pixel_buffer.to_numpy() / sample_count.to_numpy()[:, :, None]
    return buffer.swapaxes(0, 1).reshape((HEIGHT * WIDTH, 4))


def clear_data():
    DATA = None
    mesh.clear_data()
    instance.clear_data()
    material.clear_data()


def set_camera_from_matrix(self, cam_matrix, cam_angle):
    # Create a camera from a matrix for the viewport
    self.camera = Camera(None, self.image_width, self.image_height, cam_matrix, cam_angle)
