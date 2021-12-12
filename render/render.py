from .scene import Scene
import taichi as ti
from .camera import Camera
from .vector import *
from .integrator import Integrator


@ti.data_oriented
class Render:
    ''' The render object handles rendering and dealing with the framebuffer '''
    def __init__(self):
        ti.init(arch=ti.gpu)
        # scene, camera and integrator objects to be used
        self.camera: Camera = None
        self.scene: Scene = None
        self.integrator = Integrator()

    def sync_depsgraph(self, depsgraph):
        self.scene = Scene(depsgraph)
        self.camera = Camera(depsgraph.scene.camera, self.image_width, self.image_height)
        self.scene.commit()
        self.integrator.set_scene(self.scene)

    def set_camera_from_matrix(self, cam_matrix, cam_angle):
        # Create a camera from a matrix for the viewport
        self.camera = Camera(None, self.image_width, self.image_height, cam_matrix, cam_angle)

    def get_buffer(self):
        # get the framebuffer
        return self.pixel_buffer.to_numpy().swapaxes(0, 1).reshape(
            (self.image_height * self.image_width, 4))

    @ti.kernel
    def render_pass(self):
        # render a single sample on each pixel
        for i, j in self.pixel_buffer:
            s = (i + ti.random()) / (self.image_width - 1)
            t = (j + ti.random()) / (self.image_height - 1)
            ray = self.camera.get_ray(s, t)
            self.pixel_buffer[i, j] += self.integrator.trace_ray(ray, Vector4(1.0), 8)

    @ti.kernel
    def finish(self, n: ti.i32):
        # divide each pixel by the number of samples
        for i, j in self.pixel_buffer:
            self.pixel_buffer[i, j] /= n

    def set_resolution(self, width, height):
        # create an appropriately sized framebuffer
        self.image_width, self.image_height = width, height
        self.pixel_buffer = ti.Vector.field(n=4, dtype=ti.f32, shape=(width, height))

    def save(self):
        ti.imwrite(self.pixel_buffer, 'out.png')
