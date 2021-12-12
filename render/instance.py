import taichi as ti
from .hit_record import empty_hit_record
from .vector import *
import numpy as np
from .ray import *
from .mesh import mesh
from numpy.linalg import inv

# Taichi Object Instance Struct
# holds a reference to the mesh contained
# (no nested instances since blender flattens for render)
instance = ti.types.struct(box_min=Vector, box_max=Vector,
                           world_to_obj=Matrix4, obj_to_world=Matrix4,
                           mesh_ptr=mesh)


def export_instance(inst, mesh_ptr):
    ''' Creates a Taichi Instance Struct from a blender depsgraph instance '''
    bound_box = np.array(inst.object.bound_box)
    mat = np.array(inst.matrix_world, dtype=np.float32).reshape(4, 4)
    bb_vertices = [np.array([v[0], v[1], v[2], 1.0]) for v in bound_box]
    world_bb_vertices = np.array([(mat @ v)[:3] for v in bb_vertices])

    return instance(box_min=list(world_bb_vertices.min(axis=0)),
                    box_max=list(world_bb_vertices.max(axis=0)),
                    world_to_obj=list(inv(mat)), obj_to_world=list(mat), mesh_ptr=mesh_ptr)


@ti.func
def hit_instance(inst, r, t_min, t_max):
    ''' Returns if a ray hits a bounding box of an instance '''
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
    # convert a ray using a matrix
    vec_new = mat @ Vector4(vec.x, vec.y, vec.z, 1.0 if is_point else 0.0)
    return Vector(vec_new.x, vec_new.y, vec_new.z)


@ti.func
def convert_to_object_space(inst, r):
    # convert ray from world space to object
    orig = convert_space(inst.world_to_obj, r.orig, True)
    dir = convert_space(inst.world_to_obj, r.dir, False)

    return Ray(orig=orig, dir=dir.normalized(), time=r.time)


@ti.data_oriented
class InstanceCache:
    ''' Instance cache keeps track of the instances exported and their data '''
    def __init__(self):
        self.ti_data = None
        self.data = []  # a dict of instances id: instance_struct TODO find a way to hash instances

    def add(self, inst, mesh_ptr):
        # add a new instance to the cache
        self.data.append(export_instance(inst, mesh_ptr))

    def commit(self):
        ''' save the instance data to taichi data'''
        self.ti_data = instance.field(shape=len(self.data))

        # TODO this loop is slow
        for i, inst in enumerate(self.data):
            self.ti_data[i] = inst
        self.num_instances = len(self.data)

    @ti.func
    def hit(self, mesh_data, r, t_min, t_max):
        # test hit all instances and return the closest mesh that is hit

        hit_anything = False
        material_id = 0
        rec = empty_hit_record()

        i = 0
        while i < self.num_instances:
            # first check the instance bbox for hits
            inst = self.ti_data[i]
            hit_box = hit_instance(inst, r, t_min, t_max)
            if hit_box:
                # convert ray and t to object space
                r_object = convert_to_object_space(inst, r)
                p_min_obj = convert_space(inst.world_to_obj, at(r, t_min), True)
                p_max_obj = convert_space(inst.world_to_obj, at(r, t_max), True)
                t_min_obj = t_from_p(r_object, p_min_obj)
                t_max_obj = t_from_p(r_object, p_max_obj)
                # now get the mesh hit
                hit_mesh, temp_rec, temp_material_id = mesh_data.hit(inst.mesh_ptr, r_object,
                                                                     t_min_obj, t_max_obj)
                # if hit set to closest and convert back to world
                if hit_mesh:
                    hit_anything = True
                    rec = temp_rec
                    rec.p = convert_space(inst.obj_to_world, rec.p, True)
                    rec.normal = convert_space(inst.obj_to_world, rec.normal, False).normalized()
                    t_max = t_from_p(r, rec.p)
                    material_id = temp_material_id
            i += 1

        return hit_anything, rec, material_id
