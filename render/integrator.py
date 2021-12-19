import taichi as ti
from .vector import *
from .ray import Ray

INFINITY = 99999999.9


@ti.data_oriented
class Integrator:
    ''' Path tracing integrator, uses the scene to test against for ray hits.'''
    def set_scene(self, scene):
        self.scene = scene

    @ti.func
    def trace_ray(self, r, background, max_depth):
        bounces = 1
        color = Vector4(1.0)
        bounces = 1

        while bounces < max_depth:
            hit, rec, mat_id = self.scene.hit(r, 0.0001, INFINITY)
            if hit:
                emitted_color = self.scene.materials.get_emission(mat_id, r, rec)
                is_scattered, scattered_ray, attenuation = self.scene.materials.get_scattering(mat_id, r, rec)

                if is_scattered:
                    color = emitted_color + color * attenuation
                    r = scattered_ray
                    bounces += 1
                else:
                    color *= emitted_color
                    break
            elif bounces == 1:
                color = background
                break
            else:
                # make sure alpha is 1 because we hit something
                color *= background
                color.w = 1.0
                break

        return color
