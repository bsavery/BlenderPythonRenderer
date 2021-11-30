import taichi as ti
from .vector import *
import numpy as np
from . import ray

# Mesh Struct
# holds start, end index to list of triangles
mesh = ti.types.struct(start_index=ti.u32, end_index=ti.u32)


# Triangle Struct
# holds three vertex indices
triangle = ti.types.struct(p0=ti.u32, p1=ti.u32, p2=ti.u32)

vertex = Point


def get_mesh_tris(blender_mesh, offset):
    num_tris = len(blender_mesh.loop_triangles)
    data = np.zeros(num_tris * 3, dtype=np.uint32)
    blender_mesh.loop_triangles.foreach_get('vertices', data)
    return data.reshape((num_tris, 3)) + offset


def get_mesh_verts(blender_mesh):
    num_verts = len(blender_mesh.vertices)
    data = np.zeros(num_verts * 3, dtype=np.float32)
    blender_mesh.vertices.foreach_get('co', data)
    return data.reshape((num_verts, 3))


def export_mesh(blender_obj, triangle_offset, vertex_offset):
    blender_mesh = blender_obj.data
    blender_mesh.calc_loop_triangles()

    vertices = get_mesh_verts(blender_mesh)
    tris = get_mesh_tris(blender_mesh, vertex_offset)
    mesh_struct = mesh(start_index=triangle_offset, end_index=(triangle_offset + len(tris)))

    return mesh_struct, tris, vertices


@ti.func
def hit_triangle(v0, v1, v2, r, t_min, t_max):
    hit = True
    t = 0.0
    e1 = v1 - v0
    e2 = v2 - v0
    s = r.orig - v0
    s1 = r.dir.cross(e2)
    s2 = s.cross(e1)
    det = s1.dot(e1)
    if abs(det) < 0.0001:
        hit = False
    else:
        det_inv = 1.0 / det
        t = s2.dot(e2) * det_inv
        if t <= t_min or t > t_max:
            hit = False
        else:
            b1 = s1.dot(s) * det_inv
            b2 = s2.dot(r.dir) * det_inv
            b3 = 1 - b1 - b2
            if b1 < 0.0 or b1 > 1.0 or b2 < 0.0 or b2 > 1.0 or b3 < 0.0 or b3 > 1.0:
                hit = False
            else:
                p = ray.at(r, t)
                hit = True

    return hit, t
