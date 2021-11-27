import taichi as ti
from .vector import *


# Object Master Struct
object_master = ti.types.struct(center=Point, radius=ti.f32)


def export_master(master):
    return object_master(center=Point(0.0), radius=1.0)
