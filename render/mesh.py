import taichi as ti
from .vector import *
from . import ray
from .hit_record import empty_hit_record, set_face_normal
import sys

# Mesh Struct
# holds start, end index to list of triangles
mesh = ti.types.struct(start_index=ti.u32, end_index=ti.u32)


def setup_data(exported_meshes):
    ''' Creates taichi data fields from numpy arrays exported from Blender '''
    # setup the pixel buffer and inflight rays
    global tris, verts, mat_indices, meshes

    verts = Point.field(shape=exported_meshes.vert_count)
    verts.from_numpy(exported_meshes.verts)

    tris = ti.field(dtype=ti.u32, shape=(exported_meshes.tri_count, 3))
    tris.from_numpy(exported_meshes.tris)

    mat_indices = ti.field(dtype=ti.u8, shape=exported_meshes.tri_count)
    mat_indices.from_numpy(exported_meshes.mat_indices)

    meshes = mesh.field(shape=len(exported_meshes.start_indices))
    meshes.start_index.from_numpy(exported_meshes.start_indices)
    meshes.end_index.from_numpy(exported_meshes.end_indices)


def clear_data():
    global tris, verts, mat_indices, meshes
    tris = None
    verts = None
    mat_indices = None
    meshes = None


@ti.func
def hit_triangle(v0, v1, v2, r, t_min, t_max):
    ''' Intersect a ray with a triangle '''
    hit = False
    rec = empty_hit_record()

    t = 0.0
    e1 = v1 - v0
    e2 = v2 - v0
    h = r.dir.cross(e2)
    a = e1.dot(h)
    if ti.abs(a) > sys.float_info.epsilon:
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
                    rec.normal = e1.cross(e2)
    return hit, rec


@ti.func
def hit(mesh_index, r, t_min, t_max):
    # hit all tris in a mesh and return the hit record and mesh material that is hit

    hit_anything = False
    material_id = 0
    rec = empty_hit_record()

    m = meshes[mesh_index]

    i = m.start_index
    while i < m.end_index:
        v0_i, v1_i, v2_i = tris[i, 0], tris[i, 1], tris[i, 2]
        hit_tri, temp_rec = hit_triangle(verts[v0_i],
                                         verts[v1_i],
                                         verts[v2_i],
                                         r, t_min, t_max)

        if hit_tri:
            hit_anything = True
            material_id = mat_indices[i]
            rec = temp_rec
            t_max = rec.t
            set_face_normal(r, rec.normal, rec)

        i += 1

    return hit_anything, rec, material_id
