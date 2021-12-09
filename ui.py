import bpy


class RENDER_PT_properties(bpy.types.Panel):
    bl_idname = "BPR_PT_Render_Properties"
    bl_label = "BPR Samples"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'render'

    @classmethod
    def poll(cls, context):
        return context.engine == 'BPR'

    def draw(self, context):
        # You can set the property values that should be used when the user
        # presses the button in the UI.
        bpr = context.scene.bpr
        self.layout.prop(bpr, 'samples')


def register():
    bpy.utils.register_class(RENDER_PT_properties)


def unregister():
    bpy.utils.unregister_class(RENDER_PT_properties)
