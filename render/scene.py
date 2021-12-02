from .mesh import MeshCache
from .instance import InstanceCache
from .material import MaterialCache
import taichi as ti
from .vector import *
import numpy as np
from .ray import Ray
import time
from .bvh import hit_aabb, build_bvh


class Scene:
    # Scene data contains a mapping of blender objects to taichi data
    # this is used for syncing data
    meshes = MeshCache()
    instances = InstanceCache()
    materials = MaterialCache()

    def __init__(self, depsgraph):
        t_start = time.time()

        # sync meshes and any materials
        for obj in depsgraph.objects:
            if obj.type == 'MESH':
                self.meshes.add(obj, self.materials)

        for instance in depsgraph.object_instances:
            if instance.object.type == 'MESH' and instance.object in self.meshes.data:
                self.instances.add(instance, self.meshes.get_mesh(instance.object))

    def commit(self):
        ''' flatten and save the data to taichi fields '''
        self.materials.commit()
        self.meshes.commit()
        self.instances.commit()

    @ti.func
    def hit(self, r, t_min, t_max):
        color = Vector4(0.0)

        hit, material_id = self.instances.hit(self.meshes, r, t_min, t_max)
        if hit:
            color = self.materials.ti_get(material_id)

        return hit, color

    @ti.func
    def trace_camera_ray(self, r, background):
        color = background
        hit, temp_color = self.hit(r, 0.0, 99999999.0)
        if hit:
            color = temp_color
        return color


'''    
    
    @ti.func
    def hit_TLAS(self, r, t_min, t_max):
        # trace against top level acceleration structure

        hit_anything = False
        color = Vector4(0.0)

        bvh_id = 0

        # walk the bvh tree
        while bvh_id != -1:
            bvh_node = self.bvh[bvh_id]

            if bvh_node.obj_id != -1:
                # this is a leaf node, check the instance
                instance = self.object_instances[bvh_node.obj_id]
                hit_i = hit_instance(instance, r, t_min, t_max)

                if hit_i:
                    # convert ray space to object
                    r_orig = instance.matrix @ Vector4(r.orig.x, r.orig.y, r.orig.z, 1.0)
                    orig = Vector(r_orig.x, r_orig.y, r_orig.z)
                    r_dir = instance.matrix @ Vector4(r.dir.x, r.dir.y, r.dir.z, 0.0)
                    dir = Vector(r_dir.x, r_dir.y, r_dir.z)
                    r_object = Ray(orig=orig, dir=dir.normalized(), time=r.time)

                    temp_hit, temp_color, temp_t = self.hit_BLAS(instance.mesh_id, r_object, t_min, t_max)
                    if temp_hit:
                        hit_anything = True
                        t_max = temp_t
                        color = temp_color
                bvh_id = bvh_node.next_id
            else:
                if hit_aabb(bvh_node, r, t_min, t_max):
                    # visit left child next (left child will visit it's next = right)
                    bvh_id = bvh_node.left_id
                else:
                    bvh_id = bvh_node.next_id

        return hit_anything, color

    @ti.func
    def hit_BLAS(self, mesh_id, r, t_min, t_max):
        # trace against bottom level acceleration structure

        hit_anything = False
        color = Vector4(0.0)

        mesh = self.meshes[mesh_id]
        start_i, end_i = mesh.start_index, mesh.end_index

        while start_i < end_i:
            v0_i, v1_i, v2_i = self.triangles[start_i, 0], self.triangles[start_i, 1], self.triangles[start_i, 2]
            hit_tri, t = hit_triangle(self.vertices[v0_i], self.vertices[v1_i], self.vertices[v2_i], r, t_min, t_max)

            if hit_tri:
                hit_anything = True
                color = self.materials[self.material_indices[start_i]]
                t_max = t
            start_i += 1

        return hit_anything, color, t_max
'''