import taichi as ti
from .vector import *

INFINITY = 99999999.9


@ti.data_oriented
class Integrator:
    ''' Path tracing integrator, uses the scene to test against for ray hits.'''
    def set_scene(self, scene):
        self.scene = scene

    @ti.func
    def trace_ray(self, r, background, max_depth):
        bounces = 1
        color = Vector4(0.0)

        accumulated_color = Vector4(1.0)
        bounces = 1

        while bounces < max_depth:
            hit, rec, mat_id = self.scene.hit(r, 0.0001, INFINITY)
            if hit:
                # increment accumulated color and set next ray orig, dir
                accumulated_color *= self.scene.get_color(mat_id)
                target = rec.p + random_in_hemi_sphere(rec.normal)
                r.orig = rec.p
                r.dir = target - rec.p
                bounces += 1
            elif bounces == 1.0:
                color = Vector4(0.0)
                break
            else:
                color = accumulated_color * background
                break

        return color
