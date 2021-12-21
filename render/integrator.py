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
        result = Vector4(0.0)
        throughput = Vector4(1.0)
        bounces = 0
        first_hit = False

        while bounces <= max_depth:
            hit, rec, mat_id = self.scene.hit(r, 0.0001, INFINITY)
            if hit:
                if bounces == 0:
                    first_hit = True
                emitted_color = self.scene.materials.get_emission(mat_id, r, rec)
                if emitted_color.w > 0.0:
                    result += emitted_color * throughput
                    break

                wo = - r.dir.normalized()
                normal = rec.normal.normalized()
                wi = self.scene.materials.sample(mat_id, wo, normal)
                pdf = self.scene.materials.pdf(mat_id, wi, normal)

                throughput *= self.scene.materials.eval(mat_id, wi, wo, normal) / pdf

                r = Ray(orig=rec.p, dir=wi, time=r.time)
                bounces += 1
            else:
                result += background * throughput
                break

        if first_hit:
            result.w = 1.0

        return result
