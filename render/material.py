import taichi as ti
from .vector import Vector4, random_in_hemi_sphere
from .ray import Ray
import math


INV_PI = 1.0 / math.pi


# Taichi data node
DATA = None


def setup_data(exported_materials):
    ''' Creates taichi data fields from numpy arrays exported from Blender '''
    # setup the data fields
    global DATA, color_data, emission_data
    DATA = ti.root.dense(ti.i, len(exported_materials.materials))
    # material data fields
    color_data = Vector4.field()
    emission_data = Vector4.field()
    DATA.place(color_data, emission_data)

    color_data.from_numpy(exported_materials.color)
    emission_data.from_numpy(exported_materials.emission_color)


def clear_data():
    global DATA
    DATA = None


@ti.func
def get_color(i):
    return color_data[i]


@ti.func
def eval(mat_id, wi, wo, normal):
    return get_color(mat_id) * INV_PI * normal.dot(wi.normalized())


@ti.func
def sample(mat_id, wo, normal):
    return random_in_hemi_sphere(normal)


@ti.func
def pdf(mat_id, wi, normal):
    cosine = normal.dot(wi.normalized())
    if cosine < 0.0:
        cosine = 0
    return cosine * INV_PI


@ti.func
def get_emission(mat_id, r, rec):
    return emission_data[mat_id]
