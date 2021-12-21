import taichi as ti
import math

# Wrap taichi types for Vector, etc.
# This provides all the functions we need!
Vector = ti.types.vector(3, ti.f32)
Vector4 = ti.types.vector(4, ti.f32)
Matrix4 = ti.types.matrix(4, 4, ti.f32)
Color = Vector
Point = Vector

EPS = 1e-8
@ti.func
def random_in_hemi_sphere(normal):
    ''' Returns a random vector around a hemisphere centered on a normal'''
    vec = random_unit_sphere()
    if vec.dot(normal) <= 0.0:
        vec = -vec

    return vec


@ti.func
def random_unit_sphere():
    a = ti.random() * math.tau
    u = ti.Vector([ti.cos(a), ti.sin(a)])
    s = ti.random() * 2.0 - 1.0
    c = ti.sqrt(1 - s**2)
    cu = c * u
    return Vector([cu[0], cu[1], s])


@ti.func
def near_zero(vec):
   return abs(vec.x) < EPS and abs(vec.y) < EPS and abs(vec.z) < EPS