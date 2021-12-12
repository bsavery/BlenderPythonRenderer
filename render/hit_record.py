import taichi as ti
from .vector import *


# struct for recording ray hits
HitRecord = ti.types.struct(p=Point, normal=Vector, t=ti.f32, front_face=ti.i32)


@ti.func
def empty_hit_record():
    ''' Constructs an empty hit record'''
    return HitRecord(p=Point(0.0), normal=Vector(0.0), t=0.0, front_face=1)


@ti.func
def set_face_normal(r, outward_normal, rec: ti.template()):
    ''' pass in hit record by reference and set front face and normal '''
    rec.front_face = r.dir.dot(outward_normal) < 0.0
    rec.normal = outward_normal if rec.front_face == 1 else -outward_normal
