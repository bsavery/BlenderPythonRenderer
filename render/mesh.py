import taichi as ti
from .vector import *
import numpy as np
from . import ray
from .hit_record import empty_hit_record, set_face_normal

# Mesh Struct
# holds start, end index to list of triangles
mesh = ti.types.struct(start_index=ti.u32, end_index=ti.u32)


def get_mesh_tris(blender_mesh, offset):
    num_tris = len(blender_mesh.loop_triangles)
    data = np.zeros(num_tris * 3, dtype=np.uint32)
    blender_mesh.loop_triangles.foreach_get('vertices', data)
    return data.reshape((num_tris, 3)) + offset


def get_mesh_normals(blender_mesh):
    num_normals = len(blender_mesh.vertices)
    data = np.zeros(num_normals * 3, dtype=np.float32)
    blender_mesh.vertices.foreach_get('normal', data)
    return data.reshape((num_normals, 3))


def get_mesh_verts(blender_mesh):
    num_verts = len(blender_mesh.vertices)
    data = np.zeros(num_verts * 3, dtype=np.float32)
    blender_mesh.vertices.foreach_get('co', data)
    return data.reshape((num_verts, 3))


def get_material_indices(blender_mesh, material_indices):
    num_indices = len(blender_mesh.loop_triangles)
    data = np.zeros(num_indices, dtype=np.uint32)
    blender_mesh.loop_triangles.foreach_get('material_index', data)
    # convert internal indices to lookup vals
    return np.array(material_indices, dtype=np.uint32)[data]


def export_mesh(blender_obj, triangle_offset, vertex_offset, material_indices):
    blender_mesh = blender_obj.data
    blender_mesh.calc_loop_triangles()
    blender_mesh.calc_normals_split()

    vertices = get_mesh_verts(blender_mesh)
    normals = get_mesh_normals(blender_mesh)
    tris = get_mesh_tris(blender_mesh, vertex_offset)
    mat_indices = get_material_indices(blender_mesh, material_indices)
    mesh_struct = mesh(start_index=triangle_offset, end_index=(triangle_offset + len(tris)))

    return mesh_struct, tris, vertices, mat_indices, normals


@ti.func
def hit_triangle(v0, v1, v2, n0, n1, n2, r, t_min, t_max):
    hit = False
    rec = empty_hit_record()

    t = 0.0
    e1 = v1 - v0
    e2 = v2 - v0
    h = r.dir.cross(e2)
    a = e1.dot(h)
    if a > 0.0001 or a < -0.0001:
        f = 1.0 / a
        s = r.orig - v0
        u = f * s.dot(h)
        if u <= 1.0 and u >= 0.0:
            q = s.cross(e1)
            v = f * r.dir.dot(q)
            if v >= 0.0 and u + v <= 1.0:
                t = f * e2.dot(q)
                if t > t_min and t <= t_max:
                    hit = True
                    rec.p = ray.at(r, t)
                    rec.t = t
                    w = 1.0 - u - v
                    rec.normal = w * n0 + u * n1 + v * n2
                    rec.normal = rec.normal.normalized()
    return hit, rec


@ti.data_oriented
class MeshCache:
    def __init__(self):
        self.ti_data = None
        self.ti_tris = None
        self.ti_verts = None
        self.ti_mat_indices = None
        self.ti_normals = None

        self.data = {}  # a dict of blender object: mesh_struct

        self.tri_count = 0
        self.vert_count = 0

    def add(self, obj, materials):
        if obj in self.data:
            return

        material_indices = [materials.get_index(slot.material) for slot in obj.material_slots]
        if material_indices == []:
            material_indices = [0]
        mesh_struct, mesh_tris, mesh_verts, mesh_mat_indices, normals = export_mesh(obj, self.tri_count,
                                                                           self.vert_count,
                                                                           material_indices)
        if len(self.data) == 0:
            self.tris = mesh_tris
            self.verts = mesh_verts
            self.mat_indices = mesh_mat_indices
            self.normals = normals
        else:
            self.tris = np.concatenate([self.tris, mesh_tris])
            self.verts = np.concatenate([self.verts, mesh_verts])
            self.mat_indices = np.concatenate([self.mat_indices, mesh_mat_indices])
            self.normals = np.concatenate([self.normals, normals])

        self.tri_count += mesh_tris.shape[0]
        self.vert_count += mesh_verts.shape[0]
        self.data[obj] = mesh_struct

    def commit(self):
        # commit data to taichi fields and clear
        self.ti_tris = ti.field(dtype=ti.u32, shape=(self.tri_count, 3))
        self.ti_tris.from_numpy(self.tris)
        self.tris = None

        self.ti_normals = Vector.field(shape=self.vert_count)
        self.ti_normals.from_numpy(self.normals)
        self.normals = None

        self.ti_verts = Point.field(shape=self.vert_count)
        self.ti_verts.from_numpy(self.verts)
        self.vert_count = 0
        self.verts = None

        self.ti_mat_indices = ti.field(dtype=ti.u32, shape=self.tri_count)
        self.ti_mat_indices.from_numpy(self.mat_indices)
        self.tri_count = 0
        self.mat_indices = None

        self.ti_data = mesh.field(shape=len(self.data))
        i = 0
        for m in self.data.values():
            self.ti_data[i] = m
            i += 1

    @ti.func
    def hit(self, m, r, t_min, t_max):
        # hit all tris in a mesh and return the mesh material that is hit

        hit_anything = False
        material_id = 0
        rec = empty_hit_record()

        i = m.start_index
        while i < m.end_index:
            v0_i, v1_i, v2_i = self.ti_tris[i, 0], self.ti_tris[i, 1], self.ti_tris[i, 2]
            hit_tri, temp_rec = hit_triangle(self.ti_verts[v0_i],
                                             self.ti_verts[v1_i],
                                             self.ti_verts[v2_i],
                                             self.ti_normals[v0_i],
                                             self.ti_normals[v1_i],
                                             self.ti_normals[v2_i], r, t_min, t_max)

            if hit_tri:
                hit_anything = True
                material_id = self.ti_mat_indices[i]
                rec = temp_rec
                t_max = rec.t
                set_face_normal(r, rec.normal, rec)

            i += 1

        return hit_anything, rec, material_id

    def get_mesh(self, obj, materials):
        if obj not in self.data:
            self.add(obj, materials)
        return self.data[obj]
