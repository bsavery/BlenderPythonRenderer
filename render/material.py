import taichi as ti
from .vector import Vector4, random_in_hemi_sphere
import numpy as np
from .ray import Ray


@ti.data_oriented
class MaterialCache:
    ''' The Material Cache exports blender objects to Taichi arrays of data
        (Just colors for now)
    '''
    def __init__(self):
        self.ti_data = None
        self.data = []  # a list of materials index = id

    def add(self, blender_material):
        if blender_material not in self.data:
            self.data.append(blender_material)

    def commit(self):
        ''' save the material data to a temp numpy array and then taichi data'''
        color = np.array([get_color(mat) for mat in self.data], dtype=np.float32)
        emission_color = np.array([get_emission_color(mat) for mat in self.data], dtype=np.float32)
        if len(self.data) == 0:
            # fix if there is no materials
            color = np.array([[1.0, 0.0, 1.0, 1.0]], dtype=np.float32)
            emission_color = np.array([[1.0, 0.0, 1.0, 1.0]], dtype=np.float32)

        self.ti_color = Vector4.field(shape=len(color))
        self.ti_color.from_numpy(color)

        self.ti_emission = Vector4.field(shape=len(emission_color))
        self.ti_emission.from_numpy(emission_color)

    def get_index(self, blender_material):
        self.add(blender_material)
        return self.data.index(blender_material)

    @ti.func
    def get_scattering(self, mat_id, r, rec):
        out_direction = random_in_hemi_sphere(rec.normal)

        attenuation = self.ti_color[mat_id]
        do_scatter = self.ti_emission[mat_id].norm_sqr() <= 0.0000001

        return do_scatter, Ray(orig=rec.p, dir=out_direction, time=r.time), attenuation

    @ti.func
    def get_emission(self, mat_id, r, rec):
        return self.ti_emission[mat_id]


def get_color(material):
    if material is None:
        return [0.0, 0.0, 0.0, 0.0]
    else:
        return list(material.diffuse_color)


def get_emission_color(material):
    if material is not None:
        # only return the emission color for node graphs with emission nodes only.
        if material.node_tree is not None:
            nodes = material.node_tree.nodes
            for node in nodes:
                if node.bl_idname == 'ShaderNodeOutputMaterial':
                    from_node = node.inputs['Surface'].links[0].from_node
                    if from_node.bl_idname == 'ShaderNodeEmission':
                        return np.array(list(from_node.inputs['Color'].default_value), dtype=np.float32) * from_node.inputs['Strength'].default_value
    return [0.0, 0.0, 0.0, 0.0]
