'''
Render module handles exporting data from Blender to numpy and then from numpy to taichi.
In general each sub module handles it's data export and then has some taichi function for
computation.

For example mesh.py:
1.  Exports and caches meshes from the Blender scene to numpy arrays of vertices, triangles, etc
2.  Creates a taichi 'field' of the numpy arrays.  Taichi has a fast numpy->taichi pipeline.
3.  For rendering, mesh.py has the functions for intersecting a ray with triangles.

Perhaps we could separate the export and render code in the future.
'''
