"""Build and render the 3D scene of a set of posterior samples exported by `export_scene.py`.

Usage:
    blender -b --factory-startup -P blender_scene.py -- \
        --manifest products/<name>.json --outdir renders/<name> [--frames 0:100] [--save-blend]

All the physics is in the exported arrays; this script only draws them.
World frame = sky frame of `eob_dynamics.py`: X west, Y north, Z towards Earth.
The camera sits on the +Z axis (the line of sight) looking at the source, north up,
with an orthographic projection (the source is ~360 Mpc away).

Bodies are semi-transparent and everything else glows additively, so that the
picture does not depend on the drawing order of the samples. Trails fade with
the orbital phase elapsed since the body passed there.
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
GRID_HALF_EXTENT_KM = 1500.0
SPIN_ARROW_KM_PER_CHI = 200.0  # arrow protrudes from the horizon by chi * this
TRAIL_FADE_CYCLES = 0.3  # e-folding of the trail brightness, in orbits
TRAIL_RADIUS_KM = 4.0
COLOR_1 = (1.0, 0.45, 0.05)
COLOR_2 = (0.15, 0.55, 1.0)
COLOR_F = (0.9, 0.9, 0.9)


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--frames", default=None,
                   help="start:stop (python slice of frame indices) or a comma-separated list")
    p.add_argument("--resolution", default="1280x720")
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--body-alpha", type=float, default=None,
                   help="opacity of the black horizons (default: 1 for one sample, else 0.3)")
    p.add_argument("--glow", type=float, default=None,
                   help="brightness of trails, arrows and rims (default: sqrt(min(1, 4 / N)));"
                        " trails, which overlap most, scale with its square")
    p.add_argument("--save-blend", action="store_true")
    p.add_argument("--no-render", action="store_true")
    return p.parse_args(argv)


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
    mat.use_transparency_overlap = False
    mat.use_backface_culling = True
    return mat, Nodes(mat.node_tree)


def glow_material(name, color, strength, fade_attribute=False):
    """Additive emission (order independent). With `fade_attribute`, the strength is
    multiplied by exp(-(now - phase) / TRAIL_FADE_CYCLES) for now >= phase and 0 otherwise,
    `phase` being a point attribute of the geometry and `now` a property of the object."""
    mat, n = blended_material(name)
    mat.use_backface_culling = False
    out = n.new("ShaderNodeOutputMaterial")
    em = n.new("ShaderNodeEmission", Color=(*color, 1))
    if fade_attribute:
        phase = n.new("ShaderNodeAttribute", attribute_type="GEOMETRY", attribute_name="phase")
        now = n.new("ShaderNodeAttribute", attribute_type="OBJECT", attribute_name="now")
        age = n.math("SUBTRACT", now.outputs["Fac"], phase.outputs["Fac"])
        drawn = n.math("GREATER_THAN", age, 0.0)
        decay = n.math("EXPONENT", n.math("MULTIPLY", age, -1 / TRAIL_FADE_CYCLES))
        n.link(n.math("MULTIPLY", n.math("MULTIPLY", drawn, decay), strength), em.inputs["Strength"])
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


def unit_sphere_mesh(name):
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=24, radius=1.0)
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


def arrow_mesh(name, length_km, radius_km):
    """Arrow along +Z from the origin, total length `length_km`."""
    L, r = bu(length_km), bu(radius_km)
    head = min(4 * r, 0.4 * L)
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    shaft = bmesh.ops.create_cone(bm, cap_ends=True, segments=16, radius1=r, radius2=r, depth=L - head)
    bmesh.ops.translate(bm, verts=shaft["verts"], vec=(0, 0, (L - head) / 2))
    tip = bmesh.ops.create_cone(bm, cap_ends=True, segments=16, radius1=2.2 * r, radius2=0, depth=head)
    bmesh.ops.translate(bm, verts=tip["verts"], vec=(0, 0, L - head / 2))
    bm.to_mesh(mesh)
    bm.free()
    return mesh


def tube_node_group(radius_km, mat):
    """Geometry nodes turning a polyline mesh into a tube with material `mat`
    (lost in the conversion otherwise), keeping its point attributes."""
    ng = bpy.data.node_groups.new(f"tube_{mat.name}", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    n = Nodes(ng)
    gi, go = n.new("NodeGroupInput"), n.new("NodeGroupOutput")
    m2c = n.new("GeometryNodeMeshToCurve")
    circle = n.new("GeometryNodeCurvePrimitiveCircle", Resolution=6, Radius=float(bu(radius_km)))
    c2m = n.new("GeometryNodeCurveToMesh")
    set_mat = n.new("GeometryNodeSetMaterial", Material=mat)
    n.link(gi.outputs["Geometry"], m2c.inputs["Mesh"])
    n.link(m2c.outputs["Curve"], c2m.inputs["Curve"])
    n.link(circle.outputs["Curve"], c2m.inputs["Profile Curve"])
    n.link(c2m.outputs["Mesh"], set_mat.inputs["Geometry"])
    n.link(set_mat.outputs["Geometry"], go.inputs[0])
    return ng


def make_trail(name, points_km, phase, mat, node_group):
    mesh = bpy.data.meshes.new(name)
    pts = bu(points_km)
    mesh.from_pydata(pts.tolist(), [(i, i + 1) for i in range(len(pts) - 1)], [])
    mesh.attributes.new("phase", "FLOAT", "POINT").data.foreach_set("value", phase.astype(np.float32))
    obj = new_object(name, mesh, mat)
    obj.modifiers.new("tube", "NODES").node_group = node_group
    obj["now"] = 0.0
    return obj


def make_grid(normal, mat):
    n_cells = int(2 * GRID_HALF_EXTENT_KM / GRID_SPACING_KM)
    mesh = bpy.data.meshes.new("grid")
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=n_cells, y_segments=n_cells, size=bu(GRID_HALF_EXTENT_KM))
    bm.to_mesh(mesh)
    bm.free()
    obj = new_object("grid_perp_J_inf", mesh, mat)
    wire = obj.modifiers.new("wire", "WIREFRAME")
    wire.thickness = bu(2.5)
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


def keyframe_switch(obj, shown_from, shown_until, n_frames, size):
    """Scale `size` (visible) between frames [shown_from, shown_until), 0 otherwise.
    Each switch is keyed on two consecutive frames, so that the value is held between
    keys whatever the interpolation (only integer frames are rendered)."""
    switches = {shown_from, shown_until}
    keys = {0} | switches | {f - 1 for f in switches}
    for f in sorted(keys & set(range(n_frames))):
        s = size if shown_from <= f < shown_until else 0.0
        obj.scale = (s, s, s)
        obj.keyframe_insert("scale", frame=f)


def build_sample(d, mats, sphere_mesh, tube_groups, label):
    n_frames = len(d["t_det_rel"])
    merged = d["merged"]
    f_merge = int(np.argmax(merged)) if merged.any() else n_frames

    bodies = []
    for k in (1, 2):
        radius = float(d[f"radius{k}"])
        bh = new_object(f"{label}_BH{k}", sphere_mesh.copy(), mats[f"body{k}"])
        chi_mag = float(np.linalg.norm(d[f"chi{k}"][0]))
        arrow = new_object(f"{label}_spin{k}",
                           arrow_mesh(f"{label}_spin{k}", radius + chi_mag * SPIN_ARROW_KM_PER_CHI, 5.0),
                           mats[f"arrow{k}"])
        arrow.rotation_mode = "QUATERNION"
        keyframe_switch(bh, 0, f_merge, n_frames, float(bu(radius)))
        keyframe_switch(arrow, 0, f_merge, n_frames, 1.0)
        bodies.append((bh, arrow, d[f"x{k}"], d[f"chi{k}"]))
        trail = make_trail(f"{label}_trail{k}", d[f"trail{k}"], d["trail_phase"], mats[f"trail{k}"],
                           tube_groups[k - 1])
        for f in range(n_frames):
            trail["now"] = float(d["now_phase"][f])
            trail.keyframe_insert('["now"]', frame=f)

    for f in range(min(f_merge + 1, n_frames)):
        for bh, arrow, x, chi in bodies:
            bh.location = arrow.location = bu(x[f])
            bh.keyframe_insert("location", frame=f)
            arrow.keyframe_insert("location", frame=f)
            arrow.rotation_quaternion = direction_quaternion(chi[f])
            arrow.keyframe_insert("rotation_quaternion", frame=f)

    # remnant, in the frame (separation at attachment, spin x separation, spin)
    radius_f = float(d["radius_f"])
    remnant = new_object(f"{label}_remnant", sphere_mesh.copy(), mats["bodyf"])
    remnant.matrix_world = Matrix(d["remnant_basis"].tolist()).to_4x4()
    keyframe_switch(remnant, f_merge, n_frames, n_frames, float(bu(radius_f)))
    re_key, im_key = remnant_shape_keys(remnant)
    deformation = d["deformation"]
    for f in range(f_merge, n_frames):
        c = deformation[f] if abs(deformation[f]) > 1e-4 else 0
        re_key.value, im_key.value = float(np.real(c)), float(np.imag(c))
        re_key.keyframe_insert("value", frame=f)
        im_key.keyframe_insert("value", frame=f)
        if c == 0:
            break
    chi_f = d["chi_f"]
    spin_f = new_object(f"{label}_spinf",
                        arrow_mesh(f"{label}_spinf", radius_f + np.linalg.norm(chi_f) * SPIN_ARROW_KM_PER_CHI, 5.0),
                        mats["arrowf"])
    spin_f.rotation_mode = "QUATERNION"
    spin_f.rotation_quaternion = direction_quaternion(chi_f)
    keyframe_switch(spin_f, f_merge, n_frames, n_frames, 1.0)


def build(manifest_file, body_alpha, glow):
    manifest = json.load(open(manifest_file))
    scenes = Path(manifest_file).parent / "scenes"
    labels = manifest["labels"]
    n = len(labels)
    if body_alpha is None:
        body_alpha = 1.0 if n == 1 else 0.35
    if glow is None:
        glow = np.sqrt(min(1.0, 4 / n))

    scene = bpy.context.scene
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)

    world = bpy.data.worlds.new("world")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.012, 0.012, 0.02, 1)
    scene.world = world

    make_grid(manifest["J_hat_grid"], glow_material("grid", (0.35, 0.35, 0.4), 0.6))

    mats = {}
    for k, color in (("1", COLOR_1), ("2", COLOR_2), ("f", COLOR_F)):
        mats[f"body{k}"] = horizon_material(f"body{k}", color, body_alpha, 2 * glow)
        mats[f"arrow{k}"] = glow_material(f"arrow{k}", color, 3 * glow)
        mats[f"trail{k}"] = glow_material(f"trail{k}", color, 2.5 * glow**2, fade_attribute=True)
    sphere = unit_sphere_mesh("unit_sphere")
    tubes = [tube_node_group(TRAIL_RADIUS_KM, mats[f"trail{k}"]) for k in (1, 2)]

    extent = np.zeros(2)
    for label in labels:
        d = np.load(scenes / f"{label}.npz")
        build_sample(d, mats, sphere, tubes, label)
        xy = np.abs(bu(np.r_[d["trail1"], d["trail2"]][:, :2])).max(axis=0)
        margin = bu(max(float(d["radius1"]), float(d["radius2"])) + SPIN_ARROW_KM_PER_CHI)
        extent = np.maximum(extent, xy + margin)
        n_frames, fps = len(d["t_det_rel"]), int(d["fps"])

    # orthographic camera on the line of sight, looking towards the source, north up
    cam_data = bpy.data.cameras.new("camera")
    cam_data.type = "ORTHO"
    cam = new_object("camera_line_of_sight", cam_data)
    cam.location = (0, 0, bu(5 * GRID_HALF_EXTENT_KM))
    cam.rotation_euler = (0, 0, 0)
    cam_data.clip_start = 0.1
    cam_data.clip_end = bu(20 * GRID_HALF_EXTENT_KM)
    scene.camera = cam

    scene.frame_start = 0
    scene.frame_end = n_frames - 1
    scene.render.fps = fps
    return scene, cam_data, extent


def main():
    args = parse_args()
    scene, cam_data, extent = build(args.manifest, args.body_alpha, args.glow)
    w, h = map(int, args.resolution.split("x"))
    scene.render.resolution_x, scene.render.resolution_y = w, h
    scene.render.resolution_percentage = 100
    cam_data.ortho_scale = 1.1 * max(2 * extent[0], 2 * extent[1] * w / h)

    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = args.samples
    scene.view_settings.view_transform = "Standard"
    scene.render.image_settings.file_format = "PNG"

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    if args.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=str(outdir / "scene.blend"))
    if args.no_render:
        return

    if args.frames and "," in args.frames:
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
