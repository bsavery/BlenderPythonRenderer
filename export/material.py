import numpy as np


class MaterialCache:
    ''' The Material Cache exports blender objects to numpy arrays of data
        (Just colors for now)
    '''
    def __init__(self):
        self.materials = []  # a list of materials index = id

    def add(self, blender_material):
        if blender_material not in self.materials:
            self.materials.append(blender_material)

    def commit(self):
        ''' save the material data to a temp numpy array '''
        if len(self.materials) == 0:
            # fix if there is no materials
            self.color = np.array([[1.0, 0.0, 1.0, 1.0]], dtype=np.float32)
            self.emission_color = np.array([[1.0, 0.0, 1.0, 1.0]], dtype=np.float32)
        else:
            self.color = np.array([get_color(mat) for mat in self.materials], dtype=np.float32)
            self.emission_color = np.array([get_emission_color(mat) for mat in self.materials], dtype=np.float32)

    def get_index(self, blender_material):
        self.add(blender_material)
        return self.materials.index(blender_material)


def get_color(material):
    if material is None:
        return [0.0, 0.0, 0.0, 0.0]
    else:
        return list(material.diffuse_color)


def get_emission_color(material):
    col = [0.0, 0.0, 0.0, 0.0]

    if material is not None:
        # only return the emission color for node graphs with emission nodes only.
        if material.node_tree is not None:
            nodes = material.node_tree.nodes
            for node in nodes:
                if node.bl_idname == 'ShaderNodeOutputMaterial':
                    from_node = node.inputs['Surface'].links[0].from_node
                    if from_node.bl_idname == 'ShaderNodeEmission':
                        col = np.array(list(from_node.inputs['Color'].default_value), dtype=np.float32) * from_node.inputs['Strength'].default_value
    return col
