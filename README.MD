# Blender Python Renderer
A GPU renderer for prototyping and research written *purely in python*

![BMW scene render!](/examples/bmw.png "BMW Scene")

BMW scene from Mike Pan

## Installing
Download the zip of this repo and install through Blender's plugin mechanism.  Taichi will be installed automatically the first time you load.  

Alternatively, if you are a developer you probably want to manage this somewhere you edit your code and symlink the directory to your Blender's addon directory.  

## What
This is an addon for a renderer to run as an add-on to Blender.  Blender has many renderers already, including the very good Cycles renderer built in, and (shameless self-promotion of other projects I'm involved in) [Radeon ProRender](https://github.com/GPUOpen-LibrariesAndSDKs/RadeonProRenderBlenderAddon).  If you are looking to render pictures for artwork, you should use one of those.

![Cornell Box Render!](/examples/cornell_box.png "Cornell Box")

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

Some known issues:  
- Handles emissive lights and diffuse materials only
- Errors caused when doing large bounce depths and resolutions

## Contributions
Yes please! I would very much like this to be a thing that other developers contribute to, and use in their work.

## Code Layout
Repo structure:
- `__init__.py` Blender registration
- `ui.py` Panels for drawing the renderer properties in Blender (note it currently just uses Cycles samples and ray bounces)
- `engine.py` This is the class that Blender calls to execute the renderer, update the scene etc.  It passes data to everything in the `render/` directory
- `export/*` The export code.  Blender data is exported to numpy arrays. 
- `render/*` The rendering code.  Numpy arrays from `export` are passed to here and moved to taichi arrays and used for rendering on the GPU.
