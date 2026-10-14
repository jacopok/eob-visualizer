"""Build and render the 3D scene of a set of posterior samples exported by `export.py`.

Usage (or through `python -m eobviz render|scene EVENT --name <name>`):
    blender -b --factory-startup -P eobviz/blender_scene.py -- \
        --event GW250114 --name <name> [--frames 0:100] [--save-blend]
    blender -P eobviz/blender_scene.py -- --event GW250114 --name <name> --interactive
(after `python -m eobviz camera EVENT --name <name>`). `--manifest products/<name>.json`
can replace `--event` and `--name`.

--interactive opens the scene in Blender's GUI instead of rendering: scrub the timeline
or play to move back and forth in time (every quantity is keyframed on every frame), and
look through the animated camera or fly around freely. Timeline markers show the time
in the event's detector, the median merger and the camera's shot changes; a text in the
corner of the view (not rendered) shows the time of the current frame.

All the physics is in the exported arrays; this script only draws them.
World frame = sky frame of `physics.py`: X west, Y north, Z towards Earth.
The camera follows the path computed by `camera.py`; on the line of sight (+Z) it is far
away (nearly orthographic). When it leaves the line of sight, a camera icon on +Z,
connected to the centre of mass, marks the direction to Earth.

The look is set by `--style` (see STYLES): in "glow" bodies are semi-transparent and
everything else glows additively; in "solid" bodies are opaque black with coloured rims,
and the trails are thin and alpha-blended, saturating to the body colour, so that the
picture does not depend on the drawing order of the samples. "solid_highlight" adds the
max-likelihood sample in lighter colours, with longer opaque trails. Trails fade with the orbital
phase elapsed since the body passed there.

--draft is for iterating on timing and camera: Workbench (solid shading) instead of EEVEE,
plain coloured low-poly spheres and arrows, and trails cut to their last fraction of an
orbit by geometry nodes instead of fading in the shader (all the trails of a body in one
object). Frames render in ~0.2 s instead of ~5 s; the interactive view plays 200 samples at
~16 fps, and real time with `--max-samples 50`.
"""

import argparse
import json
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

KM_PER_BU = 100.0  # Blender unit = 100 km
GRID_SPACING_KM = 100.0
GRID_HALF_EXTENT_KM = 8000.0
REFERENCE_REMNANT_KM = 170.0  # GW250114; km sizes of styles and grid scale with the event
TRAIL_FADE_CYCLES = 0.3  # e-folding of the trail brightness, in orbits
COLOR_1 = (0.0331, 0.3185, 1.0)  # A #3399FF (linear)
COLOR_2 = (1.0, 0.6038, 0.0331)  # B #FFCC33
COLOR_F = (0.0331, 0.7231, 0.1845)  # C #33DD77

# Looks. body: "glow" (semi-transparent black, additive rim), "black" (opaque black,
# coloured rim) or "lit" (opaque, shaded by lights); trail: "additive" or "alpha"
# (saturates to the body colour); arrows protrude from the horizon by chi * arrow_km_per_chi;
# highlight: draw the max-likelihood sample on top of the cloud, solid and with full trails.
STYLES = {
    "glow": dict(body="glow", rim=2.0, trail="additive", trail_radius_km=4.0, trail_strength=2.5,
                 trail_cutoff=(0.5, 0.75), arrow="additive", arrow_km_per_chi=200.0,
                 arrow_radius_km=3.5, highlight=False),
    "solid": dict(body="black", rim=2.0, trail="alpha", trail_radius_km=1.5, trail_alpha=0.05,
                  trail_cutoff=(0.4, 0.6), arrow="solid", arrow_km_per_chi=400.0,
                  arrow_radius_km=2.5, highlight=False),
    "solid_highlight": dict(body="black", rim=2.0, trail="alpha", trail_radius_km=1.5, trail_alpha=0.05,
                            trail_cutoff=(0.4, 0.6), arrow="solid", arrow_km_per_chi=400.0,
                            arrow_radius_km=2.5, highlight=True, highlight_body="black",
                            highlight_rim=2.0, highlight_arrow_radius_km=4.0,
                            highlight_trail_radius_km=4.0),
    "lit": dict(body="lit", rim=0.6, trail="alpha", trail_radius_km=1.5, trail_alpha=0.05,
                trail_cutoff=(0.4, 0.6), arrow="lit", arrow_km_per_chi=400.0,
                arrow_radius_km=2.5, highlight=False),
    # a faint cloud with one solid sample in it
    "highlight": dict(body="glow", body_alpha=0.05, rim=0.4, trail="alpha", trail_radius_km=1.5,
                      trail_alpha=0.03, trail_cutoff=(0.4, 0.6), arrow="additive", arrow_strength=0.35,
                      arrow_km_per_chi=400.0, arrow_radius_km=2.0, highlight=True,
                      highlight_body="black", highlight_rim=5.0, highlight_arrow_radius_km=3.5,
                      highlight_trail_radius_km=3.5),
}


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--event", default=None, help="e.g. GW250114, with --name")
    p.add_argument("--name", default=None, help="scene set, e.g. cloud200")
    p.add_argument("--manifest", default=None, help="products/<name>.json, instead of --event and --name")
    p.add_argument("--outdir", default=None,
                   help="frames and .blend (default: <event>/renders/<name>, <name>_draft with --draft)")
    p.add_argument("--camera", default=None, help="camera path (default: products/<name>_camera.npz)")
    p.add_argument("--frames", default=None,
                   help="start:stop (python slice of frame indices) or a comma-separated list of frames")
    p.add_argument("--resolution", default="1920x810")
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--body-alpha", type=float, default=None,
                   help="opacity of the black horizons (default: 1 for one sample, else min(0.35, 15 / N))")
    p.add_argument("--glow", type=float, default=None,
                   help="brightness of trails, arrows and rims (default: min(1, 4 / N)^0.35);"
                        " trails, which overlap most, scale with its square")
    p.add_argument("--style", default="glow", choices=sorted(STYLES))
    p.add_argument("--max-samples", type=int, default=None,
                   help="show only the first N samples of the set (and the max-likelihood one)")
    p.add_argument("--draft", action="store_true",
                   help="fast, plain look (Workbench, solid colours, cut trails) for iterating")
    p.add_argument("--save-blend", action="store_true")
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--interactive", action="store_true",
                   help="set up the GUI for exploring the scene instead of rendering")
    args = p.parse_args(argv)
    if args.manifest is None:
        if not (args.event and args.name):
            p.error("give --event and --name, or --manifest")
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from eobviz.events import event_directory

        args.manifest = str(event_directory(args.event) / "products" / f"{args.name}.json")
    manifest = Path(args.manifest)
    if args.outdir is None:
        args.outdir = str(manifest.parent.parent / "renders" / (manifest.stem + ("_draft" if args.draft else "")))
    return args


def load_scene(path):
    """All the arrays of a scene file, read once (an open NpzFile re-reads an array from
    disk at every access)."""
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def bu(x_km):
    return np.asarray(x_km) / KM_PER_BU


class Nodes:
    """Small helper to build node trees."""

    def __init__(self, tree):
        self.tree = tree
        tree.nodes.clear()

    def new(self, kind, **inputs):
        node = self.tree.nodes.new(kind)
        for name, value in inputs.items():
            if hasattr(node, name):
                setattr(node, name, value)
            else:
                node.inputs[name].default_value = value
        return node

    def link(self, out, inp):
        self.tree.links.new(out, inp)

    def math(self, operation, a, b=None):
        node = self.new("ShaderNodeMath", operation=operation)
        for socket, value in zip(node.inputs, (a, b)):
            if value is None:
                continue
            if isinstance(value, (int, float)):
                socket.default_value = value
            else:
                self.link(value, socket)
        return node.outputs[0]


def blended_material(name):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.surface_render_method = "BLENDED"
    # with overlap off, EEVEE runs a depth prepass per object: faded (invisible) trail
    # segments would then hide whatever is drawn after them
    mat.use_transparency_overlap = True
    mat.use_backface_culling = True
    return mat, Nodes(mat.node_tree)


def trail_fade(n, cutoff):
    """exp(-(now - phase) / TRAIL_FADE_CYCLES) for now >= phase and 0 otherwise, smoothly
    cut off between `cutoff` (orbits); `phase` is a point attribute of the geometry and
    `now` a property of the object."""
    phase = n.new("ShaderNodeAttribute", attribute_type="GEOMETRY", attribute_name="phase")
    now = n.new("ShaderNodeAttribute", attribute_type="OBJECT", attribute_name="now")
    age = n.math("SUBTRACT", now.outputs["Fac"], phase.outputs["Fac"])
    drawn = n.math("GREATER_THAN", age, 0.0)
    decay = n.math("EXPONENT", n.math("MULTIPLY", age, -1 / TRAIL_FADE_CYCLES))
    cut = n.new("ShaderNodeMapRange", interpolation_type="SMOOTHSTEP")
    n.link(age, cut.inputs["Value"])
    for name, value in zip(("From Min", "From Max", "To Min", "To Max"), (*cutoff, 1, 0)):
        cut.inputs[name].default_value = value
    return n.math("MULTIPLY", n.math("MULTIPLY", drawn, decay), cut.outputs["Result"])


def alpha_trail_material(name, color, alpha, cutoff):
    """Alpha-blended trail of opacity alpha * fade: overlapping trails of one colour
    saturate to that colour (order independent) instead of adding up to white."""
    mat, n = blended_material(name)
    mat.use_backface_culling = False
    out = n.new("ShaderNodeOutputMaterial")
    mix = n.new("ShaderNodeMixShader")
    n.link(n.math("MULTIPLY", trail_fade(n, cutoff), alpha), mix.inputs["Fac"])
    n.link(n.new("ShaderNodeBsdfTransparent").outputs[0], mix.inputs[1])
    n.link(n.new("ShaderNodeEmission", Color=(*color, 1)).outputs[0], mix.inputs[2])
    n.link(mix.outputs[0], out.inputs["Surface"])
    return mat


def glow_material(name, color, strength, fade_cutoff=None):
    """Additive emission (order independent). With `fade_cutoff`, the strength fades
    along the trail (`trail_fade`)."""
    mat, n = blended_material(name)
    mat.use_backface_culling = False
    mat.diffuse_color = (*color, 1)  # in Workbench (--draft)
    out = n.new("ShaderNodeOutputMaterial")
    em = n.new("ShaderNodeEmission", Color=(*color, 1))
    if fade_cutoff is not None:
        n.link(n.math("MULTIPLY", trail_fade(n, fade_cutoff), strength), em.inputs["Strength"])
    else:
        em.inputs["Strength"].default_value = strength
    tr = n.new("ShaderNodeBsdfTransparent")
    add = n.new("ShaderNodeAddShader")
    n.link(tr.outputs[0], add.inputs[0])
    n.link(em.outputs[0], add.inputs[1])
    n.link(add.outputs[0], out.inputs["Surface"])
    return mat


def horizon_material(name, rim_color, alpha, rim_strength):
    """Black, partially opaque body with a glowing coloured rim."""
    mat, n = blended_material(name)
    out = n.new("ShaderNodeOutputMaterial")
    body = n.new("ShaderNodeMixShader")
    body.inputs["Fac"].default_value = alpha
    n.link(n.new("ShaderNodeBsdfTransparent").outputs[0], body.inputs[1])
    n.link(n.new("ShaderNodeEmission", Color=(0, 0, 0, 1)).outputs[0], body.inputs[2])
    lw = n.new("ShaderNodeLayerWeight", Blend=0.15)
    rim = n.new("ShaderNodeEmission", Color=(*rim_color, 1))
    n.link(n.math("MULTIPLY", n.math("POWER", lw.outputs["Facing"], 3.0), rim_strength),
           rim.inputs["Strength"])
    add = n.new("ShaderNodeAddShader")
    n.link(body.outputs[0], add.inputs[0])
    n.link(rim.outputs[0], add.inputs[1])
    n.link(add.outputs[0], out.inputs["Surface"])
    return mat


def opaque_material(name):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.surface_render_method = "DITHERED"
    return mat, Nodes(mat.node_tree)


def black_horizon_material(name, rim_color, rim_strength, rim_power=1.5):
    """Opaque black body with a coloured rim, at most as bright as the colour itself
    (brighter would clip towards white)."""
    mat, n = opaque_material(name)
    out = n.new("ShaderNodeOutputMaterial")
    lw = n.new("ShaderNodeLayerWeight", Blend=0.1)
    rim = n.new("ShaderNodeEmission", Color=(*rim_color, 1))
    strength = n.math("MULTIPLY", n.math("POWER", lw.outputs["Facing"], rim_power), rim_strength)
    strength.node.use_clamp = True
    n.link(strength, rim.inputs["Strength"])
    n.link(rim.outputs[0], out.inputs["Surface"])
    return mat


def lit_material(name, color, rim_strength, roughness=0.35, base_scale=0.35):
    """Opaque, shaded by the scene lights, with a faint coloured rim."""
    mat, n = opaque_material(name)
    out = n.new("ShaderNodeOutputMaterial")
    bsdf = n.new("ShaderNodeBsdfPrincipled", Roughness=roughness)
    bsdf.inputs["Base Color"].default_value = (*(base_scale * np.array(color)), 1)
    lw = n.new("ShaderNodeLayerWeight", Blend=0.1)
    rim = n.new("ShaderNodeEmission", Color=(*color, 1))
    n.link(n.math("MULTIPLY", n.math("POWER", lw.outputs["Facing"], 4.0), rim_strength),
           rim.inputs["Strength"])
    add = n.new("ShaderNodeAddShader")
    n.link(bsdf.outputs[0], add.inputs[0])
    n.link(rim.outputs[0], add.inputs[1])
    n.link(add.outputs[0], out.inputs["Surface"])
    return mat


def solid_emission_material(name, color, strength=1.0):
    mat, n = opaque_material(name)
    out = n.new("ShaderNodeOutputMaterial")
    n.link(n.new("ShaderNodeEmission", Color=(*color, 1), Strength=strength).outputs[0],
           out.inputs["Surface"])
    return mat


def draft_material(name, color):
    """Plain colour, for Workbench (--draft)."""
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1)
    mat.roughness = 0.6
    return mat


def add_lights():
    """Key light from the observer's upper left, a dimmer fill from behind."""
    for name, direction, energy in (("key", (-0.5, 0.6, 1.0), 4.0), ("back", (0.6, -0.3, -1.0), 1.5)):
        light = bpy.data.lights.new(name, "SUN")
        light.energy = energy
        light.angle = 0.1
        obj = new_object(name, light)
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = direction_quaternion(direction)


def unit_sphere_mesh(name, segments=48):
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=segments, v_segments=segments // 2, radius=1.0)
    for f in bm.faces:
        f.smooth = True
    bm.to_mesh(mesh)
    bm.free()
    return mesh


def new_object(name, data, mat=None):
    obj = bpy.data.objects.new(name, data)
    if mat is not None:
        obj.data.materials.append(mat)
    bpy.context.collection.objects.link(obj)
    return obj


def arrow_mesh(name, length_km, radius_km, collar_km=None):
    """Arrow along +Z from the origin, total length `length_km`; with `collar_km`, a flared
    collar where it leaves a horizon of that radius, so that it looks planted on it when
    it pierces the side facing the camera."""
    L, r = bu(length_km), bu(radius_km)
    head = min(4 * r, 0.4 * L)
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    shaft = bmesh.ops.create_cone(bm, cap_ends=True, segments=16, radius1=r, radius2=r, depth=L - head)
    bmesh.ops.translate(bm, verts=shaft["verts"], vec=(0, 0, (L - head) / 2))
    tip = bmesh.ops.create_cone(bm, cap_ends=True, segments=16, radius1=2.2 * r, radius2=0, depth=head)
    bmesh.ops.translate(bm, verts=tip["verts"], vec=(0, 0, L - head / 2))
    if collar_km is not None:
        collar = bmesh.ops.create_cone(bm, cap_ends=True, segments=24, radius1=3 * r, radius2=r, depth=2 * r)
        bmesh.ops.translate(bm, verts=collar["verts"], vec=(0, 0, bu(collar_km) + 0.5 * r))
    bm.to_mesh(mesh)
    bm.free()
    return mesh


def tube_node_group(radius_km, mat, window=None):
    """Geometry nodes turning a polyline mesh into a tube with material `mat`
    (lost in the conversion otherwise), keeping its point attributes. With `window`
    (orbits), only the points with now - window < phase <= now are kept, from their point
    attributes `shown_from` and `shown_until` (frames, `trail_window_frames`): the trail
    of --draft, which cannot fade."""
    ng = bpy.data.node_groups.new(f"tube_{mat.name}", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    n = Nodes(ng)
    gi, go = n.new("NodeGroupInput"), n.new("NodeGroupOutput")
    m2c = n.new("GeometryNodeMeshToCurve")
    circle = n.new("GeometryNodeCurvePrimitiveCircle", Resolution=6 if window is None else 4,
                   Radius=float(bu(radius_km)))
    c2m = n.new("GeometryNodeCurveToMesh")
    set_mat = n.new("GeometryNodeSetMaterial", Material=mat)
    if window is None:
        n.link(gi.outputs["Geometry"], m2c.inputs["Mesh"])
    else:
        ng["window"] = window
        frame = n.new("GeometryNodeInputSceneTime").outputs["Frame"]
        shown_from, shown_until = (n.new("GeometryNodeInputNamedAttribute", data_type="FLOAT", Name=name)
                                   .outputs["Attribute"] for name in ("shown_from", "shown_until"))
        outside = n.math("MAXIMUM", n.math("LESS_THAN", frame, shown_from),
                         n.math("LESS_THAN", n.math("SUBTRACT", shown_until, frame), 0.5))
        cut = n.new("GeometryNodeDeleteGeometry", domain="POINT")
        n.link(gi.outputs["Geometry"], cut.inputs["Geometry"])
        n.link(outside, cut.inputs["Selection"])
        n.link(cut.outputs["Geometry"], m2c.inputs["Mesh"])
    n.link(m2c.outputs["Curve"], c2m.inputs["Curve"])
    n.link(circle.outputs["Curve"], c2m.inputs["Profile Curve"])
    n.link(c2m.outputs["Mesh"], set_mat.inputs["Geometry"])
    n.link(set_mat.outputs["Geometry"], go.inputs[0])
    return ng


def trail_window_frames(phase, now_phase, window):
    """First and last+1 frames at which the trail point of orbital phase `phase` is within
    `window` orbits behind the body (now_phase, per frame, does not decrease); the number
    of frames if never."""
    return (np.searchsorted(now_phase, phase, side="left"),  # first frame with now >= phase
            np.searchsorted(now_phase, phase + window, side="left"))  # ... with now >= phase + window


def make_trail(name, points_km, phase, mat, node_group):
    mesh = bpy.data.meshes.new(name)
    pts = bu(points_km)
    mesh.from_pydata(pts.tolist(), [(i, i + 1) for i in range(len(pts) - 1)], [])
    mesh.attributes.new("phase", "FLOAT", "POINT").data.foreach_set("value", phase.astype(np.float32))
    obj = new_object(name, mesh, mat)
    obj.modifiers.new("tube", "NODES").node_group = node_group
    obj["now"] = 0.0
    return obj


def make_cut_trails(name, trails, mat, node_group):
    """The trails [(points_km, phase, now_phase)] of one body of all the samples as one
    object, cut by `node_group` (with a window): one modifier evaluation per frame
    instead of one per sample, which makes most of the cost of a --draft frame."""
    window = node_group["window"]
    pts = bu(np.concatenate([t[0] for t in trails]))
    starts = np.cumsum([0] + [len(t[0]) for t in trails])[:-1]
    edges = np.concatenate([s + np.c_[np.arange(len(t[0]) - 1), np.arange(1, len(t[0]))]
                            for s, t in zip(starts, trails)])
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(pts.tolist(), edges.tolist(), [])
    shown = [np.concatenate(f) for f in zip(*(trail_window_frames(phase, now, window) for _, phase, now in trails))]
    for attribute, frames in zip(("shown_from", "shown_until"), shown):
        mesh.attributes.new(attribute, "FLOAT", "POINT").data.foreach_set("value", frames.astype(np.float32))
    obj = new_object(name, mesh, mat)
    obj.modifiers.new("tube", "NODES").node_group = node_group
    return obj


def make_grid(normal, mat, scale=1.0):
    n_cells = int(2 * GRID_HALF_EXTENT_KM / GRID_SPACING_KM)
    mesh = bpy.data.meshes.new("grid")
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=n_cells, y_segments=n_cells, size=bu(scale * GRID_HALF_EXTENT_KM))
    bm.to_mesh(mesh)
    bm.free()
    obj = new_object("grid_perp_J_inf", mesh, mat)
    wire = obj.modifiers.new("wire", "WIREFRAME")
    wire.thickness = bu(2.5 * scale)
    wire.use_replace = True
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = direction_quaternion(normal)
    return obj


def direction_quaternion(v):
    return Vector((0, 0, 1)).rotation_difference(Vector(v).normalized())


def remnant_shape_keys(obj):
    """Shape keys of the unit sphere: r = 1 + c (x^2 - y^2) + s (2 x y)
    = 1 + Re[(c + i s) e^{-2 i phi}] sin^2 theta."""
    obj.shape_key_add(name="Basis")
    co = np.empty(3 * len(obj.data.vertices))
    obj.data.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    x, y = co[:, 0], co[:, 1]
    keys = []
    for name, pattern in (("re", x**2 - y**2), ("im", 2 * x * y)):
        key = obj.shape_key_add(name=name)
        key.data.foreach_set("co", (co * (1 + pattern[:, None])).ravel())
        key.slider_min, key.slider_max = -1, 1
        keys.append(key)
    return keys


def animate(idblock, data_path, values, frames=None):
    """Keyframes on every frame (or on `frames`), constant interpolation, written in bulk.
    `values` has shape (n_frames,) or (n_frames, n_components)."""
    values = np.asarray(values, float)
    values = values.reshape(len(values), -1)
    frames = np.arange(len(values)) if frames is None else np.asarray(frames)
    ad = idblock.animation_data or idblock.animation_data_create()
    if ad.action is None:
        ad.action = bpy.data.actions.new(f"{idblock.name}_action")
    for i in range(values.shape[1]):
        fc = ad.action.fcurve_ensure_for_datablock(idblock, data_path, index=i)
        kp = fc.keyframe_points
        kp.add(len(frames))
        kp.foreach_set("co", np.c_[frames, values[:, i]].ravel().astype(np.float32))
        kp.foreach_set("interpolation", np.zeros(len(frames), dtype=np.int32))  # CONSTANT
        fc.update()


def direction_quaternions(v):
    """Quaternions (w, x, y, z) of the shortest rotations taking +Z to the directions v."""
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    q = np.c_[1 + v[:, 2], -v[:, 1], v[:, 0], np.zeros(len(v))]
    antiparallel = q[:, 0] < 1e-9
    q[antiparallel] = [0, 1, 0, 0]
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def visibility(n_frames, shown_from, shown_until, size=1.0):
    f = np.arange(n_frames)
    s = np.where((f >= shown_from) & (f < shown_until), size, 0.0)
    return np.repeat(s[:, None], 3, axis=1)


def build_sample(d, mats, sphere_mesh, tube_groups, label, style, cut_trails=None):
    """The objects of one sample; with `cut_trails` (--draft), its trails are added to
    those lists (one per body) instead, for `make_cut_trails`."""
    n_frames = len(d["t_det_rel"])
    merged = d["merged"]
    f_merge = int(np.argmax(merged)) if merged.any() else n_frames

    for k in (1, 2):
        radius = float(d[f"radius{k}"])
        bh = new_object(f"{label}_BH{k}", sphere_mesh.copy(), mats[f"body{k}"])
        chi_mag = float(np.linalg.norm(d[f"chi{k}"][0]))
        arrow = new_object(f"{label}_spin{k}",
                           arrow_mesh(f"{label}_spin{k}", radius + chi_mag * style["arrow_km_per_chi"],
                                      style["arrow_radius_km"]),
                           mats[f"arrow{k}"])
        arrow.rotation_mode = "QUATERNION"
        animate(bh, "scale", visibility(n_frames, 0, f_merge, float(bu(radius))))
        animate(arrow, "scale", visibility(n_frames, 0, f_merge))
        animate(bh, "location", bu(d[f"x{k}"]))
        animate(arrow, "location", bu(d[f"x{k}"]))
        animate(arrow, "rotation_quaternion", direction_quaternions(d[f"chi{k}"]))
        if cut_trails is not None:
            cut_trails[k - 1].append((d[f"trail{k}"], d["trail_phase"], d["now_phase"]))
            continue
        trail = make_trail(f"{label}_trail{k}", d[f"trail{k}"], d["trail_phase"], mats[f"trail{k}"],
                           tube_groups[k - 1])
        animate(trail, '["now"]', d["now_phase"])

    # remnant, in the frame (separation at attachment, spin x separation, spin)
    radius_f = float(d["radius_f"])
    remnant = new_object(f"{label}_remnant", sphere_mesh.copy(), mats["bodyf"])
    remnant.matrix_world = Matrix(d["remnant_basis"].tolist()).to_4x4()
    animate(remnant, "scale", visibility(n_frames, f_merge, n_frames, float(bu(radius_f))))
    # recoil: the remnant moves away from the centre of mass after the attachment
    offset = d["remnant_offset"] if "remnant_offset" in d else np.zeros((n_frames, 3))
    animate(remnant, "location", bu(offset))
    remnant_shape_keys(remnant)
    deformation = np.where(np.abs(d["deformation"]) > 1e-4, d["deformation"], 0)
    animate(remnant.data.shape_keys, 'key_blocks["re"].value', deformation.real)
    animate(remnant.data.shape_keys, 'key_blocks["im"].value', deformation.imag)
    chi_f = d["chi_f"]
    spin_f = new_object(f"{label}_spinf",
                        arrow_mesh(f"{label}_spinf", radius_f + np.linalg.norm(chi_f) * style["arrow_km_per_chi"],
                                   style["arrow_radius_km"], collar_km=radius_f),
                        mats["arrowf"])
    spin_f.rotation_mode = "QUATERNION"
    spin_f.rotation_quaternion = direction_quaternion(chi_f)
    animate(spin_f, "scale", visibility(n_frames, f_merge, n_frames))
    animate(spin_f, "location", bu(offset))


def camera_icon_mesh():
    """A movie-camera pictogram of unit size, its lens pointing along -Z."""
    mesh = bpy.data.meshes.new("earth_icon")
    bm = bmesh.new()
    body = bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=(0.55, 0.9, 0.7), verts=body["verts"])
    bmesh.ops.translate(bm, vec=(0, 0, 0.35), verts=body["verts"])
    lens = bmesh.ops.create_cone(bm, cap_ends=True, segments=24, radius1=0.38, radius2=0.18, depth=0.4)
    bmesh.ops.translate(bm, vec=(0, 0, -0.2), verts=lens["verts"])
    for x in (-0.25, 0.25):
        reel = bmesh.ops.create_cone(bm, cap_ends=True, segments=24, radius1=0.3, radius2=0.3, depth=0.12)
        bmesh.ops.rotate(bm, verts=reel["verts"], cent=(0, 0, 0), matrix=Matrix.Rotation(np.pi / 2, 3, "Y"))
        bmesh.ops.translate(bm, vec=(0, x * 1.1, 1.0), verts=reel["verts"])
    bm.to_mesh(mesh)
    bm.free()
    return mesh


def make_earth_marker(track, mat):
    """Camera icon on the line of sight (+Z) with a line to the centre of mass and a
    label, scaled with the view width and shown only when the camera is off that line."""
    n_frames = len(track["width"])
    width = bu(track["width"])
    shown = track["earth_marker"]
    height = 0.33 * width  # distance of the icon from the centre of mass
    size = 0.022 * width * shown

    icon = new_object("earth_icon", camera_icon_mesh(), mat)
    animate(icon, "location", np.c_[np.zeros(n_frames), np.zeros(n_frames), height])
    animate(icon, "scale", np.repeat(size[:, None], 3, axis=1))

    line_mesh = bpy.data.meshes.new("line_of_sight")
    bm = bmesh.new()
    cyl = bmesh.ops.create_cone(bm, cap_ends=True, segments=12, radius1=1, radius2=1, depth=1)
    bmesh.ops.translate(bm, vec=(0, 0, 0.5), verts=cyl["verts"])
    bm.to_mesh(line_mesh)
    bm.free()
    line = new_object("line_of_sight", line_mesh, mat)
    r = 0.0012 * width * shown
    animate(line, "scale", np.c_[r, r, height * (shown > 0)])

    text = bpy.data.curves.new("earth_label", "FONT")
    text.body = "to Earth"
    text.align_x, text.align_y = "LEFT", "CENTER"
    label = new_object("earth_label", text, mat)
    # next to the icon, towards the right of the frame
    right = np.cross(track["up"], track["direction"])
    animate(label, "location", np.c_[np.zeros(n_frames), np.zeros(n_frames), height]
            + 0.035 * width[:, None] * right)
    animate(label, "scale", np.repeat((0.025 * width * shown)[:, None], 3, axis=1))
    return label


def make_camera(track, scene):
    cam_data = bpy.data.cameras.new("camera")
    cam_data.type = "PERSP"
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width = 36.0
    cam_data.clip_start = 0.01
    cam_data.clip_end = bu(2 * float(track["distance"].max()))
    cam = new_object("camera", cam_data)
    cam.rotation_mode = "QUATERNION"
    animate(cam, "location", bu(track["location"]))
    z = track["direction"]  # the camera looks along its local -Z
    y = track["up"]
    x = np.cross(y, z)
    quats = [np.array(Matrix(np.stack([x[f], y[f], z[f]], axis=1).tolist()).to_quaternion())
             for f in range(len(z))]
    animate(cam, "rotation_quaternion", quats)
    animate(cam_data, "lens", 36.0 / (2 * np.tan(track["fov"] / 2)))
    scene.camera = cam
    return quats


def sample_materials(style, body_alpha, glow, prefix="", strong=False):
    """Materials of one set of samples; `strong`: the highlighted sample."""
    mats = {}
    body = style.get("highlight_body", style["body"]) if strong else style["body"]
    rim = style.get("highlight_rim", style["rim"]) if strong else style["rim"]
    arrow = "solid" if strong else style["arrow"]
    for k, color in (("1", COLOR_1), ("2", COLOR_2), ("f", COLOR_F)):
        # the remnants nearly coincide across samples: their rims and arrows add up most
        g = 1.0 if strong else glow ** (2.5 if k == "f" else 1.5)
        if strong:  # lighter than the saturated cloud
            color = tuple(0.45 * np.array(color) + 0.55)
        if body == "glow":
            mats[f"body{k}"] = horizon_material(f"{prefix}body{k}", color, body_alpha, rim * g)
        elif body == "black":
            mats[f"body{k}"] = black_horizon_material(f"{prefix}body{k}", color, rim)
        else:
            mats[f"body{k}"] = lit_material(f"{prefix}body{k}", color, rim)
        if arrow == "additive":
            mats[f"arrow{k}"] = glow_material(f"{prefix}arrow{k}", color, style.get("arrow_strength", 3.0) * g)
        elif arrow == "solid":
            mats[f"arrow{k}"] = solid_emission_material(f"{prefix}arrow{k}", color, 1.5 if strong else 1.0)
        else:
            mats[f"arrow{k}"] = lit_material(f"{prefix}arrow{k}", color, 0.3, roughness=0.3, base_scale=0.8)
        if style["trail"] == "additive":
            mats[f"trail{k}"] = glow_material(f"{prefix}trail{k}", color,
                                              style["trail_strength"] * (1.0 if strong else glow**2),
                                              fade_cutoff=style["trail_cutoff"])
        elif strong:
            mats[f"trail{k}"] = alpha_trail_material(f"{prefix}trail{k}", color, 1.0, (0.75, 1.0))
        else:
            mats[f"trail{k}"] = alpha_trail_material(f"{prefix}trail{k}", color, style["trail_alpha"],
                                                     style["trail_cutoff"])
    return mats


def draft_sample_materials(prefix="", strong=False):
    """Plain colours for --draft: bodies, arrows and trails of a body share one."""
    mats = {}
    for k, color in (("1", COLOR_1), ("2", COLOR_2), ("f", COLOR_F)):
        if strong:  # lighter than the cloud
            color = tuple(0.45 * np.array(color) + 0.55)
        mats[f"body{k}"] = mats[f"arrow{k}"] = mats[f"trail{k}"] = draft_material(f"{prefix}draft{k}", color)
    return mats


def build(manifest_file, camera_file, body_alpha, glow, style, draft=False, max_samples=None):
    manifest = json.load(open(manifest_file))
    scenes = Path(manifest_file).parent / "scenes"
    infos = {label: json.load(open(scenes / f"{label}.json")) for label in manifest["labels"]}
    likelihood = lambda label: infos[label]["sample"]["log_likelihood"]
    labels = manifest["labels"][:max_samples]
    if max(infos, key=likelihood) not in labels:  # with --max-samples
        labels.append(max(infos, key=likelihood))
    n = len(labels)
    if body_alpha is None:
        body_alpha = 1.0 if n == 1 else min(0.35, 15 / n)
    if glow is None:
        glow = min(1.0, 4 / n) ** 0.35
    # length scale of the event: a round multiple of the GW250114 sizes
    r_f = np.median([float(load_scene(scenes / f"{label}.npz")["radius_f"]) for label in labels[:20]])
    scale = min((1.0, 2.0, 2.5, 5.0), key=lambda x: abs(np.log(x * REFERENCE_REMNANT_KM / r_f)))
    style = {k: v * scale if k.endswith("_km") or k.endswith("_km_per_chi") else v
             for k, v in style.items()}
    print(f"length scale {scale} (grid spacing {scale * GRID_SPACING_KM:.0f} km)")

    scene = bpy.context.scene
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)

    world = bpy.data.worlds.new("world")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.012, 0.012, 0.02, 1)
    scene.world = world

    make_grid(manifest["J_hat_grid"], glow_material("grid", (0.35, 0.35, 0.4), 0.6), scale)

    if draft:
        world.color = (0.012, 0.012, 0.02)  # the Workbench background
        mats = draft_sample_materials()
        sphere = unit_sphere_mesh("unit_sphere", segments=16)
        window = style["trail_cutoff"][1]
    else:
        if style["body"] == "lit" or style["arrow"] == "lit":
            add_lights()
        mats = sample_materials(style, style.get("body_alpha", body_alpha), glow)
        sphere = unit_sphere_mesh("unit_sphere")
        window = None
    tubes = [tube_node_group(style["trail_radius_km"], mats[f"trail{k}"], window) for k in (1, 2)]

    cut_trails = ([], []) if draft else None
    best = max(labels, key=likelihood) if style["highlight"] and n > 1 else None
    for label in labels:
        d = load_scene(scenes / f"{label}.npz")
        n_frames, fps = len(d["t_det_rel"]), int(d["fps"])
        timeline = {k: d[k] for k in ("t_det_rel", "ifo", "reference_time")}
        if label != best:
            build_sample(d, mats, sphere, tubes, label, style, cut_trails)
    if draft:
        for k in (1, 2):
            make_cut_trails(f"trails{k}", cut_trails[k - 1], mats[f"trail{k}"], tubes[k - 1])

    if best is not None:
        if draft:
            hmats = draft_sample_materials(prefix="maxL_", strong=True)
        else:
            hmats = sample_materials(style, 1.0, glow, prefix="maxL_", strong=True)
        htubes = [tube_node_group(style["highlight_trail_radius_km"], hmats[f"trail{k}"],
                                  None if window is None else 1.0) for k in (1, 2)]
        hstyle = dict(style, arrow_radius_km=style["highlight_arrow_radius_km"])
        hcut_trails = ([], []) if draft else None
        build_sample(load_scene(scenes / f"{best}.npz"), hmats, sphere, htubes, f"maxL_{best}", hstyle,
                     hcut_trails)
        if draft:
            for k in (1, 2):
                make_cut_trails(f"maxL_{best}_trail{k}", hcut_trails[k - 1], hmats[f"trail{k}"], htubes[k - 1])

    track = load_scene(camera_file)
    assert len(track["width"]) == n_frames
    camera_quaternions = make_camera(track, scene)
    label = make_earth_marker(track, glow_material("earth", (1.0, 0.85, 0.5), 1.0))
    label.rotation_mode = "QUATERNION"  # parallel to the image plane, upright
    animate(label, "rotation_quaternion", camera_quaternions)

    scene.frame_start = 0
    scene.frame_end = n_frames - 1
    scene.render.fps = fps
    t_merge = float(np.median([info["t_attach_det_rel"] for info in infos.values()]))  # of all the samples
    add_time_markers(scene, timeline, t_merge, track)
    return scene, timeline, t_merge, track


def add_time_markers(scene, timeline, t_merge, track):
    """Timeline markers: detector time every 0.1 s, the median merger, and where the camera
    leaves and comes back to the line of sight. The detector time and the time from the
    merger are also keyframed as custom properties of the scene."""
    t = timeline["t_det_rel"]
    frame_of = lambda x: int(np.argmin(np.abs(t - x)))
    for tick in np.arange(np.ceil(t[0] * 10), np.floor(t[-1] * 10) + 1) / 10:
        scene.timeline_markers.new(f"{tick:+.1f} s", frame=frame_of(tick))
    scene.timeline_markers.new("merger", frame=frame_of(t_merge))
    off_axis = track["earth_marker"] > 0
    for f in np.flatnonzero(np.diff(off_axis.astype(int))):
        scene.timeline_markers.new("camera off axis" if off_axis[f + 1] else "camera on axis", frame=int(f + 1))
    scene["t_det_rel"] = 0.0
    scene["t_minus_merger"] = 0.0
    animate(scene, '["t_det_rel"]', t)
    animate(scene, '["t_minus_merger"]', t - t_merge)


def make_clock(scene, timeline, t_merge, material):
    """A text in the lower left corner of the camera view with the time of the current
    frame, updated on frame change; not rendered."""
    text = bpy.data.curves.new("clock", "FONT")
    text.align_x, text.align_y = "LEFT", "BOTTOM"
    clock = new_object("clock", text, material)
    clock.hide_render = True
    clock.parent = scene.camera
    t = timeline["t_det_rel"]
    ifo, ref = str(timeline["ifo"]), float(timeline["reference_time"])
    aspect = scene.render.resolution_x / scene.render.resolution_y

    def update(scene, depsgraph=None):
        f = min(max(scene.frame_current, 0), len(t) - 1)
        text.body = (f"{ifo} time - {ref:.1f} s: {t[f]:+.4f} s\n"
                     f"from merger: {(t[f] - t_merge) * 1e3:+.1f} ms\nframe {scene.frame_current}")
        cam = scene.camera.data
        depth = 5 * cam.clip_start
        half_width = depth * cam.sensor_width / 2 / cam.lens
        clock.location = (-0.96 * half_width, -0.94 * half_width / aspect, -depth)
        clock.scale = (0.035 * half_width,) * 3

    update(scene)
    bpy.app.handlers.frame_change_post.append(update)


def setup_gui(scene, track, draft=False):
    """Look through the scene camera, with clipping that fits the scene and a rendered
    view (solid with --draft); play back dropping frames to keep real time."""
    clip_end = float(bu(4 * track["distance"].max()))
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type != "VIEW_3D":
                continue
            space = area.spaces.active
            space.clip_start, space.clip_end = 0.01, clip_end
            if draft:
                space.shading.type = "SOLID"
                space.shading.color_type = "MATERIAL"
                space.shading.background_type = "WORLD"
            else:
                space.shading.type = "RENDERED"
            space.region_3d.view_perspective = "CAMERA"
    try:
        scene.sync_mode = "FRAME_DROP"
    except (AttributeError, TypeError):
        pass
    scene.frame_set(scene.frame_start)


def main():
    args = parse_args()
    camera = args.camera or str(Path(args.manifest).with_name(Path(args.manifest).stem + "_camera.npz"))
    scene, timeline, t_merge, track = build(args.manifest, camera, args.body_alpha, args.glow,
                                            STYLES[args.style], args.draft, args.max_samples)
    w, h = map(int, args.resolution.split("x"))
    scene.render.resolution_x, scene.render.resolution_y = w, h
    scene.render.resolution_percentage = 100

    if args.draft:
        scene.render.engine = "BLENDER_WORKBENCH"
        scene.display.shading.light = "STUDIO"
        scene.display.shading.color_type = "MATERIAL"
        scene.display.render_aa = "FXAA"
    else:
        scene.render.engine = "BLENDER_EEVEE"
        scene.eevee.taa_render_samples = args.samples
    scene.view_settings.view_transform = "Standard"
    scene.render.image_settings.file_format = "PNG"

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    if args.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=str(outdir / "scene.blend"))
    if args.interactive:
        make_clock(scene, timeline, t_merge, glow_material("clock", (1.0, 0.85, 0.5), 1.0))
        if not bpy.app.background:
            # once the window is up
            bpy.app.timers.register(lambda: setup_gui(scene, track, args.draft), first_interval=0.1)
        return
    if args.no_render:
        return

    if args.frames and ":" not in args.frames:
        for f in map(int, args.frames.split(",")):
            scene.frame_set(f)
            scene.render.filepath = str(outdir / f"3d_{f:04d}.png")
            bpy.ops.render.render(write_still=True)
        return
    if args.frames:
        frames = range(scene.frame_start, scene.frame_end + 1)
        a, b = args.frames.split(":")
        frames = frames[slice(int(a) if a else None, int(b) if b else None)]
        scene.frame_start, scene.frame_end = frames.start, frames.stop - 1
        scene.frame_step = frames.step
    scene.render.filepath = str(outdir / "3d_####")
    bpy.ops.render.render(animation=True)


main()
