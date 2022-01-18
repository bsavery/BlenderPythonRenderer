import numpy as np
from numpy.linalg import inv


def export_instance(inst):
    ''' Exports Instance data from a blender depsgraph instance '''
    bound_box = np.array(inst.object.bound_box, dtype=np.float32)
    mat = np.array(inst.matrix_world, dtype=np.float32).reshape(4, 4)
    bb_vertices = [np.array([v[0], v[1], v[2], 1.0]) for v in bound_box]
    world_bb_vertices = np.array([(mat @ v)[:3] for v in bb_vertices], dtype=np.float32)

    box_min = world_bb_vertices.min(axis=0)
    box_max = world_bb_vertices.max(axis=0)
    world_to_obj = inv(mat)

    return box_min, box_max, world_to_obj, mat


class InstanceCache:
    ''' Instance cache keeps track of the instances exported and their data '''
    def __init__(self):
        self.instance_count = 0

    def add(self, inst, mesh_id):
        # add a new instance to the cache
        box_min, box_max, world_to_obj, obj_to_world = export_instance(inst)
        if self.instance_count == 0:
            self.box_min = np.array([box_min], dtype=np.float32)
            self.box_max = np.array([box_max], dtype=np.float32)
            self.world_to_obj = np.array([world_to_obj], dtype=np.float32)
            self.obj_to_world = np.array([obj_to_world], dtype=np.float32)
            self.mesh_id = np.array([mesh_id], dtype=np.uint8)
        else:
            self.box_min = np.concatenate([self.box_min, [box_min]])
            self.box_max = np.concatenate([self.box_max, [box_max]])
            self.world_to_obj = np.concatenate([self.world_to_obj, [world_to_obj]])
            self.obj_to_world = np.concatenate([self.obj_to_world, [obj_to_world]])
            self.mesh_id = np.concatenate([self.mesh_id, np.array([mesh_id], dtype=np.uint8)])

        self.instance_count += 1

    def commit(self):
        ''' save the instance data to taichi data'''
        self.box_min = np.array(self.box_min, dtype=np.float32)
        self.box_max = np.array(self.box_max, dtype=np.float32)
        self.world_to_obj = np.array(self.world_to_obj, dtype=np.float32)
        self.obj_to_world = np.array(self.world_to_obj, dtype=np.float32)
        self.mesh_id = np.array(self.mesh_id, dtype=np.uint8)
