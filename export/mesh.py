import numpy as np


def get_mesh_tris(blender_mesh, offset):
    ''' Returns numpy array of vertex indices, + offset'''
    num_tris = len(blender_mesh.loop_triangles)
    data = np.zeros(num_tris * 3, dtype=np.uint32)
    blender_mesh.loop_triangles.foreach_get('vertices', data)
    return data.reshape((num_tris, 3)) + offset


def get_mesh_normals(blender_mesh, offset):
    ''' Returns a numpy array of normals one for each face, with offset
        TODO add per vertex normals?
    '''
    num_normals = len(blender_mesh.loop_triangles)
    data = np.zeros(num_normals * 3, dtype=np.uint32)
    blender_mesh.loop_triangles.foreach_get('normal', data)
    return data.reshape((num_normals, 3)) + offset


def get_mesh_verts(blender_mesh):
    ''' Returns a numpy array of vertex positions
        TODO maybe we could de-duplicate data
    '''
    num_verts = len(blender_mesh.vertices)
    data = np.zeros(num_verts * 3, dtype=np.float32)
    blender_mesh.vertices.foreach_get('co', data)
    return data.reshape((num_verts, 3))


def get_material_indices(blender_mesh, material_indices):
    ''' Gets a numpy array of face material indices '''
    num_indices = len(blender_mesh.loop_triangles)
    data = np.zeros(num_indices, dtype=np.uint32)
    blender_mesh.loop_triangles.foreach_get('material_index', data)
    # convert internal indices to lookup vals
    return np.array(material_indices, dtype=np.uint32)[data]


def export_mesh(blender_obj, triangle_offset, vertex_offset, material_indices):
    ''' Gets numpy arrays of the mesh data '''
    blender_mesh = blender_obj.data
    blender_mesh.calc_loop_triangles()

    vertices = get_mesh_verts(blender_mesh)
    tris = get_mesh_tris(blender_mesh, vertex_offset)
    mat_indices = get_material_indices(blender_mesh, material_indices)
    # normals = get_mesh_normals(blender_mesh, triangle_offset)
    mesh_struct = (triangle_offset, (triangle_offset + len(tris)))

    return mesh_struct, tris, vertices, mat_indices  # , normals


class MeshCache:
    ''' Caches all the mesh data and a list of structs describing meshes '''
    def __init__(self):
        self.data = {}  # a dict of blender object: mesh_struct

        self.tri_count = 0
        self.vert_count = 0
        self.mesh_count = 0

    def add(self, obj, materials):
        ''' Add an object mesh to the cache '''
        if obj.name_full in self.data.keys():
            return

        material_indices = [materials.get_index(slot.material) for slot in obj.material_slots]
        if material_indices == []:
            # if no materials assigned set all to material 0
            material_indices = [0]
        mesh_struct, mesh_tris, mesh_verts, mesh_mat_indices = export_mesh(obj,
                                                                           self.tri_count,
                                                                           self.vert_count,
                                                                           material_indices)
        start_index, end_index = mesh_struct

        # copy the exported mesh data to the arrays
        if self.mesh_count == 0:
            self.tris = mesh_tris
            self.verts = mesh_verts
            self.mat_indices = mesh_mat_indices
            self.start_indices = np.array([start_index], dtype=np.uint32)
            self.end_indices = np.array([end_index], dtype=np.uint32)
            # self.normals = normals
        else:
            self.tris = np.concatenate([self.tris, mesh_tris])
            self.verts = np.concatenate([self.verts, mesh_verts])
            self.mat_indices = np.concatenate([self.mat_indices, mesh_mat_indices])
            self.start_indices = np.concatenate([self.start_indices, np.array([start_index], dtype=np.uint32)])
            self.end_indices = np.concatenate([self.end_indices, np.array([end_index], dtype=np.uint32)])
            # self.normals = np.concatenate([self.normals, normals])

        self.tri_count += mesh_tris.shape[0]
        self.vert_count += mesh_verts.shape[0]
        self.data[obj.name_full] = self.mesh_count
        self.mesh_count += 1

    def get_mesh(self, obj, materials):
        if obj.name_full not in self.data.keys():
            self.add(obj, materials)
        return self.data[obj.name_full]
