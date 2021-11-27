from .object import object_master, export_master
from .instance import object_instance, export_instance, hit_instance
import taichi as ti


class World:
    def __init__(self, depsgraph):
        # create a list of object masters
        num_masters = len([o for o in depsgraph.objects if o.type == 'MESH'])
        num_instances = len([o for o in depsgraph.object_instances if o.object.type == 'MESH'])

        self.object_masters = object_master.field(shape=num_masters)
        # this will hold a dictionary of objects to id's
        master_dict = {}
        i = 0
        for master in depsgraph.objects:
            if master.type == 'MESH':
                self.object_masters[i] = export_master(master)
                master_dict[master] = i
                i += 1

        # create a list of object instances
        self.object_instances = object_instance.field(shape=num_instances)
        i = 0
        for instance in depsgraph.object_instances:
            if instance.object.type == 'MESH':
                self.object_instances[i] = export_instance(instance, 0)
                i += 1

    @ti.func
    def hit(self, r, t_min, t_max):
        hit_anything = False

        for i in range(self.object_instances.shape[0]):
            if hit_instance(self.object_instances[i], r, t_min, t_max):
                hit_anything = True

        return hit_anything
