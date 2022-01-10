from .scene import Scene
import taichi as ti
from .camera import Camera
from .vector import *
from .integrator import Integrator
from .ray import Ray


InFlightRay = ti.types.struct(depth=ti.i32, ray=Ray, throughput=Vector4)


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
        # get the framebuffer and divide by sample count
        buffer = self.pixel_buffer.to_numpy() / self.sample_count.to_numpy()[:,:,None]
        return buffer.swapaxes(0, 1).reshape((self.image_height * self.image_width, 4))

    @ti.kernel
    def render_pass(self) -> ti.i32:
        ''' for each pixel in the image we
            1.  Check if it needs a new camera ray
            2.  Trace one more bounce
            3.  check if need to stop, accumulate, etc
            Note that we use taichi's bit mask function to stop rendering pixels
            that have all samples done.

            RETURNS num sampls completed
        '''
        samples_done = 0
        for i, j in self.pixel_buffer:
            if self.sample_count[i, j] == self.num_samples:
                # skip this pixel
                continue

            inflight = self.rays_in_flight[i, j]
            ray = inflight.ray
            depth = inflight.depth
            throughput = inflight.throughput

            if depth == 0:
                # add a new ray for pixel that needs new one
                s = (i + ti.random()) / (self.image_width - 1)
                t = (j + ti.random()) / (self.image_height - 1)
                ray = self.camera.get_ray(s, t)
                depth = self.max_depth
                throughput = Vector4(1.0)

            # integrate one bounce along path
            ray, throughput, ray_stop = self.integrator.trace_ray(ray, throughput, Vector4(0.0))
            depth -= 1

            # reset inflight for this pixel if stopped
            if ray_stop or depth == 0:
                # somewhat hack to set alpha to 1 if more than one bounce in
                if self.max_depth - inflight.depth > 0:
                    throughput.w = 1.0

                # add accumulated color to pixel
                self.pixel_buffer[i, j] += throughput
                self.sample_count[i, j] += 1
                samples_done += 1
                depth = 0

            self.rays_in_flight[i, j] = InFlightRay(depth=depth, ray=ray, throughput=throughput)

        return samples_done

    def set_resolution(self, width, height):
        # create an appropriately sized framebuffer
        self.image_width, self.image_height = width, height

    def setup_render(self, num_samples, max_depth):
        self.max_depth = max_depth
        self.num_samples = num_samples
        self.pixel_buffer = ti.Vector.field(n=4, dtype=ti.f32,
                                            shape=(self.image_width, self.image_height))
        self.sample_count = ti.field(dtype=ti.i32, shape=(self.image_width, self.image_height))

        # setup a buffer of pixels to trace one for each pixel
        self.rays_in_flight = InFlightRay.field(shape=(self.image_width, self.image_height))

    def save(self):
        ti.imwrite(self.pixel_buffer, 'out.png')
