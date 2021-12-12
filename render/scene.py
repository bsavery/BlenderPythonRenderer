from .mesh import MeshCache
from .instance import InstanceCache
from .material import MaterialCache
import taichi as ti
from .vector import *
import numpy as np
from .ray import Ray
import time


class Scene:
    ''' Scene data of meshes, instances, and materials'''
    def __init__(self, depsgraph):
        t_start = time.time()
        # Scene data contains a mapping of blender objects to taichi data
        # this is used for syncing data
        self.meshes = MeshCache()
        self.instances = InstanceCache()
        self.materials = MaterialCache()

        # sync meshes and any materials
        for obj in depsgraph.objects:
            if obj.type == 'MESH':
                self.meshes.add(obj, self.materials)

        # sync instances
        for instance in depsgraph.object_instances:
            if instance.object.type == 'MESH':
                self.instances.add(instance, self.meshes.get_mesh(instance.object, self.materials))

    def commit(self):
        ''' flatten and save the data to taichi fields '''
        self.materials.commit()
        self.meshes.commit()
        self.instances.commit()

    @ti.func
    def hit(self, r, t_min, t_max):
        ''' Returns the hit point and record of a ray against the scene '''
        return self.instances.hit(self.meshes, r, t_min, t_max)

    @ti.func
    def get_color(self, material_id):
        return self.materials.ti_get(material_id)
