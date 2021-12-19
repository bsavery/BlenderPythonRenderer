import bpy
import bgl
from .render.render import Render
import numpy as np
import time


class CustomRenderEngine(bpy.types.RenderEngine):
    # These three members are used by blender to set up the
    # RenderEngine; define its internal name, visible name and capabilities.
    bl_idname = "BPR"
    bl_label = "BPR"
    bl_use_preview = True

    # Init is called whenever a new render engine instance is created. Multiple
    # instances may exist at the same time, for example for a viewport and final
    # render.
    def __init__(self):
        self.scene_data = None
        self.draw_data = None
        self.renderer: Render = None

    # When the render engine instance is destroy, this is called. Clean up any
    # render engine data here, for example stopping running render threads.
    def __del__(self):
        pass

    def update(self, data=None, depsgraph=None):
        t = time.time()
        self.renderer = Render()
        scene = depsgraph.scene
        scale = scene.render.resolution_percentage / 100.0
        self.resolution = (int(scene.render.resolution_x * scale),
                           int(scene.render.resolution_y * scale))
        self.renderer.set_resolution(self.resolution[0], self.resolution[1])

        self.renderer.sync_depsgraph(depsgraph)
        print("Total export", time.time() - t)

    # This is the method called by Blender for both final renders (F12) and
    # small preview for materials, world and lights.
    def render(self, depsgraph):
        # render the number of samples
        scene = depsgraph.scene
        # result = self.begin_result(0, 0, self.size_x, self.size_y)

        # layer = result.layers[0].passes["Combined"]

        num_samples = scene.cycles.samples
        max_bounces = scene.cycles.max_bounces
        t = time.time()
        n = 0
        while n < num_samples:
            if self.test_break():
                break
            self.renderer.render_pass(max_bounces)
            n += 1
            # result = self.begin_result(0, 0, self.size_x, self.size_y)
            # layer = result.layers[0].passes["Combined"]
            self.rect = self.renderer.get_buffer() / n
            # self.end_result(result)
            self.update_progress(n / num_samples)

        self.renderer.finish(n)

        result = self.begin_result(0, 0, self.resolution[0], self.resolution[1])
        layer = result.layers[0].passes["Combined"]
        layer.rect = self.renderer.get_buffer()
        self.end_result(result)

        print("Total render", time.time() - t)
        # self.renderer.save()

    # For viewport renders, this method gets called once at the start and
    # whenever the scene or 3D viewport changes. This method is where data
    # should be read from Blender in the same thread. Typically a render
    # thread will be started to do the work while keeping Blender responsive.
    def view_update(self, context, depsgraph):
        self.update(depsgraph=depsgraph)
        dimensions = context.region.width, context.region.height
        self.resolution = dimensions
        self.renderer.set_resolution(*dimensions)
        # this angle needs to be fixed
        self.cam_matrix = context.region_data.view_matrix.inverted()
        self.renderer.set_camera_from_matrix(self.cam_matrix, 72.0)
        self.rect = do_render(16, self.renderer)

    # For viewport renders, this method is called whenever Blender redraws
    # the 3D viewport. The renderer is expected to quickly draw the render
    # with OpenGL, and not perform other expensive work.
    # Blender will draw overlays for selection and editing on top of the
    # rendered image automatically.
    def view_draw(self, context, depsgraph):
        region = context.region
        scene = depsgraph.scene

        if context.region_data.view_matrix.inverted() != self.cam_matrix:
            self.view_update(context, depsgraph)

        # Get viewport dimensions
        dimensions = region.width, region.height

        # Bind shader that converts from scene linear to display space,
        bgl.glEnable(bgl.GL_BLEND)
        bgl.glBlendFunc(bgl.GL_ONE, bgl.GL_ONE_MINUS_SRC_ALPHA)
        self.bind_display_space_shader(scene)

        if not self.draw_data or self.draw_data.dimensions != dimensions:
            self.draw_data = CustomDrawData(dimensions)
        self.draw_data.set_texture(self.rect, self.resolution)

        self.draw_data.draw()

        self.unbind_display_space_shader()
        bgl.glDisable(bgl.GL_BLEND)


class CustomDrawData:
    def __init__(self, window_dimensions):
        # Generate dummy float image buffer
        self.dimensions = window_dimensions
        width, height = window_dimensions

        pixels = [0.0, 0.0, 0.0, 0.0] * width * height
        pixels = bgl.Buffer(bgl.GL_FLOAT, width * height * 4, pixels)

        # Generate texture
        self.texture = bgl.Buffer(bgl.GL_INT, 1)
        bgl.glGenTextures(1, self.texture)
        bgl.glActiveTexture(bgl.GL_TEXTURE0)
        bgl.glBindTexture(bgl.GL_TEXTURE_2D, self.texture[0])
        bgl.glTexImage2D(bgl.GL_TEXTURE_2D, 0, bgl.GL_RGBA16F,
                         width, height, 0, bgl.GL_RGBA, bgl.GL_FLOAT, pixels)
        bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_MIN_FILTER, bgl.GL_LINEAR)
        bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_MAG_FILTER, bgl.GL_LINEAR)
        bgl.glBindTexture(bgl.GL_TEXTURE_2D, 0)

        # Bind shader that converts from scene linear to display space,
        # use the scene's color management settings.
        shader_program = bgl.Buffer(bgl.GL_INT, 1)
        bgl.glGetIntegerv(bgl.GL_CURRENT_PROGRAM, shader_program)

        # Generate vertex array
        self.vertex_array = bgl.Buffer(bgl.GL_INT, 1)
        bgl.glGenVertexArrays(1, self.vertex_array)
        bgl.glBindVertexArray(self.vertex_array[0])

        texturecoord_location = bgl.glGetAttribLocation(shader_program[0], "texCoord")
        position_location = bgl.glGetAttribLocation(shader_program[0], "pos")

        bgl.glEnableVertexAttribArray(texturecoord_location)
        bgl.glEnableVertexAttribArray(position_location)

        # Generate geometry buffers for drawing textured quad
        position = [0.0, 0.0, width, 0.0, width, height, 0.0, height]
        position = bgl.Buffer(bgl.GL_FLOAT, len(position), position)
        texcoord = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
        texcoord = bgl.Buffer(bgl.GL_FLOAT, len(texcoord), texcoord)

        self.vertex_buffer = bgl.Buffer(bgl.GL_INT, 2)

        bgl.glGenBuffers(2, self.vertex_buffer)
        bgl.glBindBuffer(bgl.GL_ARRAY_BUFFER, self.vertex_buffer[0])
        bgl.glBufferData(bgl.GL_ARRAY_BUFFER, 32, position, bgl.GL_STATIC_DRAW)
        bgl.glVertexAttribPointer(position_location, 2, bgl.GL_FLOAT, bgl.GL_FALSE, 0, None)

        bgl.glBindBuffer(bgl.GL_ARRAY_BUFFER, self.vertex_buffer[1])
        bgl.glBufferData(bgl.GL_ARRAY_BUFFER, 32, texcoord, bgl.GL_STATIC_DRAW)
        bgl.glVertexAttribPointer(texturecoord_location, 2, bgl.GL_FLOAT, bgl.GL_FALSE, 0, None)

        bgl.glBindBuffer(bgl.GL_ARRAY_BUFFER, 0)
        bgl.glBindVertexArray(0)

    def set_texture(self, pixels, dimensions):
        # Generate texture
        width, height = dimensions

        pixels = pixels.flatten()
        buffer = bgl.Buffer(bgl.GL_FLOAT, width * height * 4, pixels.flatten())

        bgl.glActiveTexture(bgl.GL_TEXTURE0)
        bgl.glBindTexture(bgl.GL_TEXTURE_2D, self.texture[0])
        bgl.glTexImage2D(bgl.GL_TEXTURE_2D, 0, bgl.GL_RGBA16F,
                         width, height, 0, bgl.GL_RGBA, bgl.GL_FLOAT, buffer)
        bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_MIN_FILTER, bgl.GL_LINEAR)
        bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_MAG_FILTER, bgl.GL_LINEAR)
        bgl.glBindTexture(bgl.GL_TEXTURE_2D, 0)

    def __del__(self):
        bgl.glDeleteBuffers(2, self.vertex_buffer)
        bgl.glDeleteVertexArrays(1, self.vertex_array)
        bgl.glBindTexture(bgl.GL_TEXTURE_2D, 0)
        bgl.glDeleteTextures(1, self.texture)

    def draw(self):
        bgl.glActiveTexture(bgl.GL_TEXTURE0)
        bgl.glBindTexture(bgl.GL_TEXTURE_2D, self.texture[0])
        bgl.glBindVertexArray(self.vertex_array[0])
        bgl.glDrawArrays(bgl.GL_TRIANGLE_FAN, 0, 4)
        bgl.glBindVertexArray(0)
        bgl.glBindTexture(bgl.GL_TEXTURE_2D, 0)


# RenderEngines also need to tell UI Panels that they are compatible with.
# We recommend to enable all panels marked as BLENDER_RENDER, and then
# exclude any panels that are replaced by custom panels registered by the
# render engine, or that are not supported.
def get_panels():
    exclude_panels = {
        'VIEWLAYER_PT_filter',
        'VIEWLAYER_PT_layer_passes',
    }

    panels = []
    for panel in bpy.types.Panel.__subclasses__():
        if hasattr(panel, 'COMPAT_ENGINES') and 'BLENDER_RENDER' in panel.COMPAT_ENGINES:
            if panel.__name__ not in exclude_panels:
                panels.append(panel)

    return panels


def register():
    # Register the RenderEngine
    bpy.utils.register_class(CustomRenderEngine)

    for panel in get_panels():
        panel.COMPAT_ENGINES.add('BPR')


def unregister():
    bpy.utils.unregister_class(CustomRenderEngine)

    for panel in get_panels():
        if 'PYTHON' in panel.COMPAT_ENGINES:
            panel.COMPAT_ENGINES.remove('BPR')


def do_render(num_samples, renderer):
    n = 0
    while n < num_samples:
        renderer.render_pass()
        n += 1
    return renderer.get_buffer() / n
