import taichi as ti
from .vector import *
import numpy as np
from .ray import *
from .mesh import mesh
from .bvh import *

# Object Instance Struct
instance = ti.types.struct(box_min=Vector, box_max=Vector, matrix=Matrix4, mesh_ptr=mesh)


def export_instance(inst, mesh_ptr):
    bound_box = np.array(inst.object.bound_box)
    # add column to multiply nice
    bound_box = np.hstack((bound_box, np.ones((8, 1))))
    # multiply by matrix
    bound_box = bound_box @ np.array(inst.matrix_world).T
    # remove column
    bound_box = np.delete(bound_box, -1, 1)

    matrix = [list(row) for row in inst.matrix_world.inverted()]

    return instance(box_min=list(bound_box.min(axis=0)),
                    box_max=list(bound_box.max(axis=0)),
                    matrix=matrix, mesh_ptr=mesh_ptr)


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
def convert_to_object_space(inst, r):
    # convert ray space to object
    r_orig = inst.matrix @ Vector4(r.orig.x, r.orig.y, r.orig.z, 1.0)
    orig = Vector(r_orig.x, r_orig.y, r_orig.z)
    r_dir = inst.matrix @ Vector4(r.dir.x, r.dir.y, r.dir.z, 0.0)
    dir = Vector(r_dir.x, r_dir.y, r_dir.z)

    return Ray(orig=orig, dir=dir.normalized(), time=r.time)


@ti.data_oriented
class InstanceCache:
    ti_data = None
    data = []  # a dict of instances id: instance_struct  #TODO find a way to hash instances

    def add(self, inst, mesh_ptr):
        self.data.append(export_instance(inst, mesh_ptr))

    def commit(self):
        ''' save the instance data to taichi data'''
        self.ti_data = instance.field(shape=len(self.data))

        for i, inst in enumerate(self.data):
            self.ti_data[i] = inst

        self.ti_bvh = build_bvh(self.data)

    @ti.func
    def ti_get(self, i):
        return self.ti_data[i]

    @ti.func
    def hit(self, mesh_data, r, t_min, t_max):
        # hit all instances and return the mesh material that is hit

        hit_anything = False
        material_id = 0

        bvh_id = 0

        # walk the bvh tree
        while bvh_id != -1:
            bvh_node = self.ti_bvh[bvh_id]

            if bvh_node.obj_id != -1:
                # this is a leaf node, check the instance
                inst = self.ti_data[bvh_node.obj_id]
                # convert ray space to object
                r_object = convert_to_object_space(inst, r)

                hit_mesh, t, mat_id = mesh_data.hit(inst.mesh_ptr, r_object, t_min, t_max)
                # if hit set to closest
                if hit_mesh:
                    hit_anything = True
                    t_max = t
                    material_id = mat_id
                bvh_id = bvh_node.next_id
            else:
                if hit_aabb(bvh_node, r, t_min, t_max):
                    # visit left child next (left child will visit it's next = right)
                    bvh_id = bvh_node.left_id
                else:
                    bvh_id = bvh_node.next_id

        return hit_anything, material_id
