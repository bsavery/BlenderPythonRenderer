import taichi as ti

from .hit_record import empty_hit_record
from .vector import *
import numpy as np
from .ray import *
from .mesh import mesh
from mathutils import Vector as b_vec

# Object Instance Struct
instance = ti.types.struct(box_min=Vector, box_max=Vector, matrix=Matrix4, matrix_world=Matrix4, mesh_ptr=mesh)


def export_instance(inst, mesh_ptr):
    
    bound_box = np.array(inst.object.bound_box)
    mat = inst.matrix_world if inst.is_instance else inst.object.matrix_world
    bb_vertices = [b_vec(v) for v in bound_box]
    world_bb_vertices = np.array([mat @ v for v in bb_vertices])

    matrix = [list(row) for row in mat.inverted()]
    matrix_world = [list(row) for row in mat]

    return instance(box_min=list(world_bb_vertices.min(axis=0)),
                    box_max=list(world_bb_vertices.max(axis=0)),
                    matrix=matrix, matrix_world=matrix_world, mesh_ptr=mesh_ptr)


@ti.func
def hit_instance(inst, r, t_min, t_max):
    intersect = True
    min_aabb, max_aabb = inst.box_min, inst.box_max
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


@ti.func
def convert_space(mat, vec, is_point):
    vec_new = mat @ Vector4(vec.x, vec.y, vec.z, 1.0 if is_point else 0.0)
    return Vector(vec_new.x, vec_new.y, vec_new.z)


@ti.func
def convert_to_object_space(inst, r):
    # convert ray space to object
    orig = convert_space(inst.matrix, r.orig, True)
    dir = convert_space(inst.matrix, r.dir, False)

    return Ray(orig=orig, dir=dir.normalized(), time=r.time)


@ti.func
def convert_t(r_orig, r_dest, t):
    P = at(r_orig, t)
    return (P - r_dest.orig)[0] / r_dest.dir[0]


@ti.data_oriented
class InstanceCache:
    def __init__(self):
        self.ti_data = None
        self.data = []  # a dict of instances id: instance_struct  #TODO find a way to hash instances

    def add(self, inst, mesh_ptr):
        self.data.append(export_instance(inst, mesh_ptr))

    def commit(self):
        ''' save the instance data to taichi data'''
        self.ti_data = instance.field(shape=len(self.data))

        for i, inst in enumerate(self.data):
            self.ti_data[i] = inst
        self.num_instances = len(self.data)

    @ti.func
    def ti_get(self, i):
        return self.ti_data[i]

    @ti.func
    def hit(self, mesh_data, r, t_min, t_max):
        # hit all instances and return the mesh material that is hit

        hit_anything = False
        material_id = 0
        rec = empty_hit_record()

        i = 0
        while i < self.num_instances:
            # first check the instance bbox for hits
            inst = self.ti_get(i)
            hit_box = hit_instance(inst, r, t_min, t_max)
            if hit_box:
                # convert ray and t to object space
                r_object = convert_to_object_space(inst, r)
                # now get the mesh hit
                hit_mesh, temp_rec, temp_material_id = mesh_data.hit(inst.mesh_ptr, r_object, t_min, t_max)
                # if hit set to closest
                if hit_mesh:
                    hit_anything = True
                    t_max = temp_rec.t
                    rec = temp_rec
                    rec.p = convert_space(inst.matrix_world, rec.p, True)
                    rec.normal = convert_space(inst.matrix_world, rec.normal, False)
                    material_id = temp_material_id
            i += 1

        return hit_anything, rec, material_id
