import taichi as ti
from .vector import *
from .ray import Ray
from . import instance
from . import material

INFINITY = 99999999.9


@ti.func
def trace_ray(r, throughput, background):
    ray_stop = False

    hit, rec, mat_id = instance.hit(r, 0.001, INFINITY)
    if hit:
        emitted_color = material.get_emission(mat_id, r, rec) * throughput
        if emitted_color.w > 0.0:
            throughput *= emitted_color
            ray_stop = True
        else:
            wo = - r.dir.normalized()
            normal = rec.normal.normalized()
            wi = material.sample(mat_id, wo, normal)
            pdf = material.pdf(mat_id, wi, normal)

            throughput *= material.eval(mat_id, wi, wo, normal) / pdf
            # stop if throughput is small
            if throughput.x < 0.0001 and throughput.y < 0.0001 and throughput.z < 0.0001:
                ray_stop = True
            r = Ray(orig=rec.p, dir=wi, time=r.time)
    else:
        throughput *= background
        ray_stop = True

    return r, throughput, ray_stop
