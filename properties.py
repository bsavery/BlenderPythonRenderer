import bpy


class BPRRenderSettings(bpy.types.PropertyGroup):
    samples: bpy.props.IntProperty(default=16)


def register():
    bpy.utils.register_class(BPRRenderSettings)
    bpy.types.Scene.bpr = bpy.props.PointerProperty(type=BPRRenderSettings)


def unregister():
    bpy.utils.unregister_class(BPRRenderSettings)
