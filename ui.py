import bpy


# UI Panels for displaying propertis of this addon


class RENDER_PT_properties(bpy.types.Panel):
    bl_idname = "BPR_PT_Render_Properties"
    bl_label = "Samples"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'render'

    @classmethod
    def poll(cls, context):
        # only show this banel if BPR is selected
        return context.engine == 'BPR'

    def draw(self, context):
        cycles_settings = context.scene.cycles
        self.layout.prop(cycles_settings, 'samples')
        self.layout.prop(cycles_settings, 'max_bounces')


def register():
    bpy.utils.register_class(RENDER_PT_properties)


def unregister():
    bpy.utils.unregister_class(RENDER_PT_properties)
