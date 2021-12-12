import bpy


# UI Panels for displaying propertis of this addon


class RENDER_PT_properties(bpy.types.Panel):
    bl_idname = "BPR_PT_Render_Properties"
    bl_label = "BPR Samples"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'render'

    @classmethod
    def poll(cls, context):
        # only show this banel if BPR is selected
        return context.engine == 'BPR'

    def draw(self, context):
        bpr = context.scene.bpr
        self.layout.prop(bpr, 'samples')


def register():
    bpy.utils.register_class(RENDER_PT_properties)


def unregister():
    bpy.utils.unregister_class(RENDER_PT_properties)
