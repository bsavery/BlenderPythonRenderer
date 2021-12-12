import bpy


class BPRRenderSettings(bpy.types.PropertyGroup):
    ''' For now we just track a samples property '''
    samples: bpy.props.IntProperty(default=16)


def register():
    bpy.utils.register_class(BPRRenderSettings)
    # attach the property group to the scene
    bpy.types.Scene.bpr = bpy.props.PointerProperty(type=BPRRenderSettings)


def unregister():
    bpy.utils.unregister_class(BPRRenderSettings)
