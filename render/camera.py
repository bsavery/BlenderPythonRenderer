import taichi as ti
from .vector import *
from .ray import Ray
import math


# data fields
origin = Vector(0.0)
lower_left_corner = Vector(0.0)
horizontal = Vector(0.0)
vertical = Vector(0.0)
u = Vector(0.0)
v = Vector(0.0)


def setup_data(exported_camera):
    global origin, lower_left_corner, horizontal, vertical, u, v
    origin = Vector(exported_camera.origin)
    lower_left_corner = Vector(exported_camera.lower_left_corner)
    horizontal = Vector(exported_camera.horizontal)
    vertical = Vector(exported_camera.vertical)
    u = Vector(exported_camera.u)
    v = Vector(exported_camera.v)


@ti.func
def get_ray(s, t):
    ''' Computes random sample based on st of image space '''
    rd = ti.Vector([0.0, 0.0])
    offset = u * rd.x + v * rd.y
    return Ray(orig=(origin + offset),
               dir=(lower_left_corner + s*horizontal
                    + t*vertical - origin - offset),
               time=ti.random())
