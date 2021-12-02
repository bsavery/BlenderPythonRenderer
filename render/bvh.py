import taichi as ti
import copy
import random
from .vector import *
import numpy as np

# struct for bvh node
# holds bvh box bounds and either the object reference or ids of other bvh nodes
# for left, right, next in tree
BVHNode = ti.types.struct(box_min=Point, box_max=Point,
                          obj_id=ti.i32,
                          left_id=ti.i32, right_id=ti.i32,
                          parent_id=ti.i32, next_id=ti.i32)


def surrounding_box(box1, box2):
    ''' Calculates the surround bbox of two bboxes '''
    box1_min, box1_max = box1
    box2_min, box2_max = box2

    small = ti.min(box1_min, box2_min)
    big = ti.max(box1_max, box2_max)
    return small, big


def get_bounding_box(obj):
    return obj.box_min, obj.box_max


def get_center(obj):
    box_min, box_max = get_bounding_box(obj)
    return (box_min + box_max) / 2.0


def sort_obj_list(obj_list):
    ''' Sort the list of objects along the longest directional span '''
    def get_x(e):
        obj = e
        return get_center(obj).x

    def get_y(e):
        obj = e
        return get_center(obj).y

    def get_z(e):
        obj = e
        return get_center(obj).z

    centers = [get_center(obj) for obj in obj_list]
    min_center = [
        min([center[0] for center in centers]),
        min([center[1] for center in centers]),
        min([center[2] for center in centers])
    ]
    max_center = [
        max([center[0] for center in centers]),
        max([center[1] for center in centers]),
        max([center[2] for center in centers])
    ]
    span_x, span_y, span_z = (max_center[0] - min_center[0],
                              max_center[1] - min_center[1],
                              max_center[2] - min_center[2])
    if span_x >= span_y and span_x >= span_z:
        obj_list.sort(key=get_x)
    elif span_y >= span_z:
        obj_list.sort(key=get_y)
    else:
        obj_list.sort(key=get_z)
    return obj_list


def build_bvh_inner(unsorted_nodes, parent_id=-1, curr_id=0):
    node_list = []

    span = len(unsorted_nodes)
    if span == 1:
        # one obj, set to obj bbox
        obj = unsorted_nodes[0]
        obj.parent_id = parent_id
        node_list = [obj]

    else:
        # sort list of object and divide and conquer
        sorted_list = sort_obj_list(unsorted_nodes)
        mid = int(span / 2)

        # pass correct start indices to sublists generation
        left_nodelist = build_bvh_inner(sorted_list[:mid], curr_id, curr_id + 1)
        right_nodelist = build_bvh_inner(sorted_list[mid:], curr_id, curr_id + len(left_nodelist) + 1)

        box_min, box_max = surrounding_box(
            (left_nodelist[0].box_min, left_nodelist[0].box_max),
            (right_nodelist[0].box_min, right_nodelist[0].box_max))

        # create this node
        node_list.append(
            BVHNode(box_min=box_min, box_max=box_max,
                    obj_id=-1,
                    left_id=curr_id + 1, right_id=curr_id + len(left_nodelist) + 1,
                    parent_id=parent_id, next_id=-1))
        # add right and left
        node_list = node_list + left_nodelist + right_nodelist

    return node_list


def set_next_id_links(bvh_node_list):
    ''' given a list of nodes set the 'next_id' link in the nodes '''
    def inner_loop(node_id):
        node = bvh_node_list[node_id]
        if node.parent_id == -1:
            return -1

        parent = bvh_node_list[node.parent_id]
        if parent.right_id != -1 and parent.right_id != node_id:
            return parent.right_id
        else:
            return inner_loop(node.parent_id)

    for i, node in enumerate(bvh_node_list):
        node.next_id = inner_loop(i)


def build_bvh(instance_list):
    ''' building function. Compress the object list to structure'''
    # create a list of bvh_nodes to sort
    print('here', len(instance_list))
    bvh_node_list = []
    for i, inst in enumerate(instance_list):
        bvh_node_list.append(BVHNode(box_min=inst.box_min, box_max=inst.box_max,
                                     obj_id=i,
                                     left_id=-1, right_id=-1,
                                     parent_id=0, next_id=-1))

    print('about to start')
    # construct temp list of node structs
    bvh_node_list = build_bvh_inner(bvh_node_list)
    set_next_id_links(bvh_node_list)
    bvh_field = BVHNode.field(shape=(len(bvh_node_list),))

    for i, node in enumerate(bvh_node_list):
        bvh_field[i] = node
    return bvh_field

def get_tri_box(verts):
    return verts.min(axis=0), verts.max(axis=0)

def get_mesh_bvh_nodes(mesh_tris, mesh_verts, id_offset, bvh_offset):
    # create a list of bvh_nodes to sort
    bvh_node_list = []
    for i, tri in enumerate(mesh_tris):
        verts = np.array([mesh_verts[tri[0]], mesh_verts[tri[1]], mesh_verts[tri[2]]])
        box_min, box_max = get_tri_box(verts)
        bvh_node_list.append(BVHNode(box_min=Point(box_min), box_max=Point(box_max),
                                     obj_id=i+id_offset,
                                     left_id=-1, right_id=-1,
                                     parent_id=0, next_id=-1))

    # construct temp list of node structs
    bvh_node_list = build_bvh_inner(bvh_node_list)
    set_next_id_links(bvh_node_list)

    for i, node in enumerate(bvh_node_list):
        if node.left_id != -1:
            node.left_id += bvh_offset
        if node.right_id != -1:
            node.right_id += bvh_offset
        if node.next_id != -1:
            node.next_id += bvh_offset
    
    return bvh_node_list


@ti.func
def hit_aabb(bvh_node, r, t_min, t_max):
    intersect = True
    min_aabb, max_aabb = bvh_node.box_min, bvh_node.box_max
    ray_direction, ray_origin = r.dir, r.orig

    for i in ti.static(range(3)):
        if ray_direction[i] == 0:
            if ray_origin[i] < min_aabb[i] or ray_origin[i] > max_aabb[i]:
                intersect = False
        else:
            i1 = (min_aabb[i] - ray_origin[i]) / ray_direction[i]
            i2 = (max_aabb[i] - ray_origin[i]) / ray_direction[i]

            new_t_max = ti.max(i1, i2)
            new_t_min = ti.min(i1, i2)

            t_max = ti.min(new_t_max, t_max)
            t_min = ti.max(new_t_min, t_min)

    if t_min > t_max:
        intersect = False
    return intersect
