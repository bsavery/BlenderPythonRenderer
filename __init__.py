# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
from . import engine
from . import properties
from . import ui

bl_info = {
    "name": "BlenderPythonRender",
    "author": "Brian Savery",
    "description": "",
    "blender": (3, 0, 0),
    "version": (0, 0, 1),
    "location": "",
    "warning": "",
    "category": "Render"
}


def check_for_taichi():
    import sys
    if 'taichi' not in sys.modules:
        import pip
        pip.main(['install', 'taichi', '--user'])


def register():
    check_for_taichi()
    engine.register()
    properties.register()
    ui.register()


def unregister():
    engine.unregister()
    ui.unregister()
    properties.unregister()
