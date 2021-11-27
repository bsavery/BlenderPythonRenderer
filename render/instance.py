import taichi as ti
from .vector import *
import numpy as np

# Object Instance Struct
object_instance = ti.types.struct(box_min=Vector, box_max=Vector, master_id=ti.i32)


def export_instance(instance, master_id):
    bound_box = np.array(instance.object.bound_box)
    # add column to multiply nice
    bound_box = np.hstack((bound_box, np.ones((8, 1))))
    # multiply by matrix
    bound_box = bound_box @ np.array(instance.matrix_world).T
    # remove column
    bound_box = np.delete(bound_box, -1, 1)

    return object_instance(box_min=list(bound_box.min(axis=0)),
                           box_max=list(bound_box.max(axis=0)),
                           master_id=master_id)


@ti.func
def hit_instance(instance, r, t_min, t_max):
    intersect = True
    min_aabb, max_aabb = instance.box_min, instance.box_max
    ray_direction, ray_origin = r.dir, r.orig

    for i in ti.static(range(3)):
        if ray_direction[i] == 0:
            if ray_origin[i] < min_aabb[i] or ray_origin[i] > max_aabb[i]:
                intersect = False
        else:
            i1 = (min_aabb[i] - ray_origin[i]) / ray_direction[i]
            i2 = (max_aabb[i] - ray_origin[i]) / ray_direction[i]

            new_t_max = ti.max(i1, i2)
            new_t_min = ti.min(i1, i2)

            t_max = ti.min(new_t_max, t_max)
            t_min = ti.max(new_t_min, t_min)

    if t_min > t_max:
        intersect = False
    return intersect
