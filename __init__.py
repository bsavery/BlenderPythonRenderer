from . import engine
from . import ui


bl_info = {
    "name": "BlenderPythonRender",
    "author": "Brian Savery",
    "description": "Blender Python Render is a render addon \
    that renders on the GPU via a python module called 'Taichi'",
    "blender": (3, 0, 0),
    "version": (0, 0, 1),
    "location": "",
    "warning": "",
    "category": "Render"
}


def check_for_taichi():
    # look for taichi and install via PIP if needed
    try:
        import taichi
    except:
        import pip
        pip.main(['install', 'taichi', '--user'])


def register():
    check_for_taichi()
    engine.register()
    ui.register()


def unregister():
    engine.unregister()
    ui.unregister()
