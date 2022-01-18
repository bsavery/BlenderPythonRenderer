import numpy as np
import math


class Camera:
    ''' Camera class '''
    def __init__(self, blender_cam, width, height, matrix=None, angle=None):
        # create the camera vectors from the data
        # note that we can override the camera matrix for viewport rendering
        aspect_ratio = width / height
        t0, t1 = 0.0, 1.0
        focus_dist = 10.0
        aperture = 0.0

        theta = blender_cam.data.angle / aspect_ratio if angle is None else angle / aspect_ratio
        h = math.tan(theta/2.0)

        cam_mat = blender_cam.matrix_world if matrix is None else matrix
        look_from = np.array([cam_mat[0][3], cam_mat[1][3], cam_mat[2][3]])
        self.u = np.array([cam_mat[0][0], cam_mat[1][0], cam_mat[2][0]])
        self.v = np.array([cam_mat[0][1], cam_mat[1][1], cam_mat[2][1]])
        w = np.array([cam_mat[0][2], cam_mat[1][2], cam_mat[2][2]])

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

    def get_data(self):
        return self.u, self.v, self.origin, self.horizontal, self.vertical, self.lower_left_corner
