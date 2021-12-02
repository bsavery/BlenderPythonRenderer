import taichi as ti
from .vector import Vector4
import numpy as np


@ti.data_oriented
class MaterialCache:
    ti_data = None
    data = []  # a list of materials index = id

    def add(self, blender_material):
        if blender_material not in self.data:
            self.data.append(blender_material)

    def commit(self):
        ''' save the material data to a temp numpy array and then taichi data'''
        self.ti_data = Vector4.field(shape=len(self.data))
        self.ti_data.from_numpy(np.array([export_material(mat) for mat in self.data], dtype=np.float32))

    def get_index(self, blender_material):
        self.add(blender_material)
        return self.data.index(blender_material)

    @ti.func
    def ti_get(self, i):
        return self.ti_data[i]


def export_material(material):
    if material is None:
        return [0.0, 0.0, 0.0, 0.0]
    else:
        return list(material.diffuse_color)
