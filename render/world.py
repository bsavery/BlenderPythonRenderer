from .mesh import mesh, triangle, vertex, export_mesh, hit_triangle
from .instance import object_instance, export_instance, hit_instance
import taichi as ti
from .vector import *
import numpy as np
from .ray import Ray


class World:
    def __init__(self, depsgraph):
        # create a list of object masters
        num_meshes = len([o for o in depsgraph.objects if o.type == 'MESH'])
        num_instances = len([o for o in depsgraph.object_instances if o.object.type == 'MESH'])

        self.meshes = mesh.field(shape=num_meshes)
        # this will hold a dictionary of objects to id's
        mesh_id_dict = {}
        i = 0
        tri_count = 0
        vert_count = 0
        tris = []
        verts = []
        for obj in depsgraph.objects:
            if obj.type == 'MESH':
                self.meshes[i], mesh_tris, mesh_verts = export_mesh(obj, tri_count, vert_count)
                mesh_id_dict[obj] = i
                i += 1

                tris = mesh_tris if tris == [] else np.concatenate([tris, mesh_tris])
                tri_count += mesh_tris.shape[0]

                verts = mesh_verts if verts == [] else np.concatenate([verts, mesh_verts])
                vert_count += mesh_verts.shape[0]

        self.triangles = ti.field(dtype=ti.u32, shape=(tri_count, 3))
        self.triangles.from_numpy(tris)

        self.vertices = vertex.field(shape=vert_count)
        self.vertices.from_numpy(verts)

        # create a list of object instances
        self.object_instances = object_instance.field(shape=num_instances)
        i = 0
        for instance in depsgraph.object_instances:
            if instance.object.type == 'MESH' and instance.object in mesh_id_dict:
                self.object_instances[i] = export_instance(instance, mesh_id_dict[instance.object], Vector4(instance.object.color))
                i += 1

    @ti.func
    def hit(self, r, t_min, t_max):
        hit_anything = False
        color = Vector4(0.0)

        for i in range(self.object_instances.shape[0]):
            # first check the instance for hits
            instance = self.object_instances[i]
            hit_box = hit_instance(instance, r, t_min, t_max)
            if hit_box:
                # check mesh for hit
                mesh = self.meshes[instance.mesh_id]
                start_i, end_i = mesh.start_index, mesh.end_index

                # convert ray space to object
                r_orig = instance.matrix @ Vector4(r.orig.x, r.orig.y, r.orig.z, 1.0)
                orig = Vector(r_orig.x, r_orig.y, r_orig.z)
                r_dir = instance.matrix @ Vector4(r.dir.x, r.dir.y, r.dir.z, 0.0)
                dir = Vector(r_dir.x, r_dir.y, r_dir.z)
                r_object = Ray(orig=orig, dir=dir.normalized(), time=r.time)

                while start_i < end_i:
                    v0_i, v1_i, v2_i = self.triangles[start_i, 0], self.triangles[start_i, 1], self.triangles[start_i, 2]
                    hit_tri, t = hit_triangle(self.vertices[v0_i], self.vertices[v1_i], self.vertices[v2_i], r_object, t_min, t_max)

                    if hit_tri:
                        hit_anything = True
                        color = instance.color
                        t_max = t
                    start_i += 1

        return hit_anything, color
