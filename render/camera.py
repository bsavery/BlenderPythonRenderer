import taichi as ti
from .vector import *
from .ray import Ray
import math


@ti.data_oriented
class Camera:
    ''' Camera class '''
    def __init__(self, blender_cam, width, height):
        aspect_ratio = width / height
        t0, t1 = 0.0, 1.0
        focus_dist = 10.0
        aperture = 0.0

        theta = blender_cam.data.angle
        h = math.tan(theta/2.0)

        cam_mat = blender_cam.matrix_world
        look_from = Vector(cam_mat[0][3], cam_mat[1][3], cam_mat[2][3])
        self.u = Vector(cam_mat[0][0], cam_mat[1][0], cam_mat[2][0])
        self.v = Vector(cam_mat[0][1], cam_mat[1][1], cam_mat[2][1])
        w = Vector(cam_mat[0][2], cam_mat[1][2], cam_mat[2][2])

        # camera position and orientation
        viewport_height = 2.0 * h
        viewport_width = aspect_ratio * viewport_height

        self.origin = look_from
        self.horizontal = focus_dist * viewport_width * self.u
        self.vertical = focus_dist * viewport_height * self.v
        self.lower_left_corner = self.origin - self.horizontal/2.0 - \
            self.vertical/2.0 - focus_dist * w

        self.lens_radius = aperture / 2.0
        self.t0, self.t1 = t0, t1

    @ti.func
    def get_ray(self, s, t):
        ''' Computes random sample based on st'''
        rd = ti.Vector([0.0, 0.0])
        offset = self.u * rd.x + self.v * rd.y
        return Ray(orig=(self.origin + offset),
                   dir=(self.lower_left_corner + s*self.horizontal
                        + t*self.vertical - self.origin - offset),
                   time=ti.random() * (self.t1 - self.t0) + self.t0)
