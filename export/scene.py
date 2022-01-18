from .mesh import MeshCache
from .instance import InstanceCache
from .material import MaterialCache
from .camera import Camera


class Scene:
    ''' Scene data of meshes, instances, and materials'''
    def __init__(self, depsgraph, resolution):
        # Scene data contains a mapping of blender objects to numpy data
        # this is used for syncing data
        self.meshes = MeshCache()
        self.instances = InstanceCache()
        self.materials = MaterialCache()
        self.camera = Camera(depsgraph.scene.camera, resolution[0], resolution[1])

        # sync meshes and any materials
        for obj in depsgraph.objects:
            if obj.type == 'MESH':
                self.meshes.add(obj, self.materials)

        # sync instances
        for instance in depsgraph.object_instances:
            if instance.object.type == 'MESH':
                self.instances.add(instance, self.meshes.get_mesh(instance.object, self.materials))

        # commit materials
        self.materials.commit()

    def free(self):
        ''' clear any memory '''
        self.meshes = MeshCache()
        self.instances = InstanceCache()
        self.materials = MaterialCache()
