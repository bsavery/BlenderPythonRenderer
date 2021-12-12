import taichi as ti
from .vector import *


# a ray class
Ray = ti.types.struct(orig=Point, dir=Vector, time=ti.f32)


@ti.func
def at(r, t):
    ''' Computes the point of ray at t '''
    return r.orig + r.dir * t


@ti.func
def t_from_p(r, P):
    ''' returns the t from P '''
    return (P[0] - r.orig[0]) / r.dir[0]
