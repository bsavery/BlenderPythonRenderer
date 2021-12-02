from .scene import Scene
import taichi as ti
from .camera import Camera
from .vector import *
from .integrator import Integrator


@ti.data_oriented
class Render:
    camera: Camera
    scene: Scene

    def __init__(self):
        ti.init(arch=ti.gpu)
    
    def sync_depsgraph(self, depsgraph):
        self.scene = Scene(depsgraph)
        self.camera = Camera(depsgraph.scene.camera, self.image_width, self.image_height)
        self.scene.commit()

    def get_buffer(self):
        return self.pixel_buffer.to_numpy().swapaxes(0, 1).reshape(
            (self.image_height * self.image_width, 4))

    @ti.kernel
    def render_pass(self):
        for i, j in self.pixel_buffer:
            s = (i + ti.random()) / (self.image_width - 1)
            t = (j + ti.random()) / (self.image_height - 1)
            ray = self.camera.get_ray(s, t)
            self.pixel_buffer[i, j] += self.scene.trace_camera_ray(ray, Vector4(0.0))

    @ti.kernel
    def finish(self, n: ti.i32):
        for i, j in self.pixel_buffer:
            self.pixel_buffer[i, j] /= n

    def set_resolution(self, width, height):
        self.image_width, self.image_height = width, height
        self.pixel_buffer = ti.Vector.field(n=4, dtype=ti.f32, shape=(width, height))

    def save(self):
        ti.imwrite(self.pixel_buffer, 'out.png')
