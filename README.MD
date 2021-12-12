# Blender Python Renderer
A GPU renderer for prototyping and research written *purely in python*

## What
This is an addon for a renderer to run as an add-on to Blender.  Blender has many renderers already, including the very good Cycles renderer built in, and (shameless self-promotion of other projects I'm involved in) [Radeon ProRender](https://github.com/GPUOpen-LibrariesAndSDKs/RadeonProRenderBlenderAddon).  If you are looking to render pictures for artwork, you should use one of those.

## Why
It seems half the repos I see on github are someone's "Toy Renderer" project. Do we really need another one?

The difference with this renderer is it is targeted at the user who is interested developing rendering code.  This allows you to:
- Write GPU rendering code in python (quickly).  While this won't be as efficient as using something like GLSL, it will be much quicker to develop.
- Use Blender to generate the scene or use one of thousands of free blender scenes
- Have a simple to read and extensible framework for your rendering code. You should see influences from Peter Shirley's [Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html) and [PBRT](https://www.pbrt.org) here.  In fact beore doing this I wrote an implementation of [Ray Tracing in One Weekend in Python](https://github.com/bsavery/ray-tracing-one-weekend-taichi)

## How
The biggest question here is executing python code on the GPU. All credit there has to go to [Taichi](https://github.com/taichi-dev/taichi) which is a very simple way to execute python code on various devices.  It's not perfect (yet) but as a Python developer I found it very easy to pick up.

## What now
Currently the renderer will load and path trace scenes in Blender when you hit the render button (F12).  Basic Blender material settings and lights are supported.  My plan is to do plenty of [optimizations](https://github.com/bsavery/BlenderPythonRenderer/milestone/2) (there's not even a BVH yet), and then tackle the big issue of full node-based materials.

## Contributions
Yes please! I would very much like this to be a thing that other developers contribute to, and use in their work.

## Code Layout
Repo structure:
- `__init__.py` Blender registration
- `properties.py` Settings for the renderer.  Perhaps in future this could use the Cycles properties?
- `ui.py` Panels for drawing the properties in Blender
- `engine.py` This is the class that Blender calls to execute the renderer, update the scene etc.  It passes data to everything in the `render/` directory
- `render/*` The rendering and export code.  I tried to make it so the mesh export and rendering code is encapsulated in `mesh.py` for example. It might make sense to separate export and rendering code in the future.  
Each data type: Instance, Mesh, Material etc caches all the data to a big numpy list which is exported to Taichi for rendering on the GPU. 
