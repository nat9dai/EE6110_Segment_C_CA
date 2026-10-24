"""Convert Google Scanned Objects (GSO) meshes to Isaac-ready USD files, and load the catalog.

Objects are scaled to fit 6.5-22 cm and re-centred on their visual bbox centre (z-up, metres).
"""

from __future__ import annotations

import glob
import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "gso_raw" / "models"
USD_DIR = DATA_DIR / "gso_usd"
POINTS_DIR = DATA_DIR / "gso_points"
CATALOG_PATH = DATA_DIR / "gso_catalog.json"
VALIDATION_PATH = DATA_DIR / "gso_validation.json"  # written by scripts/validate_gso_isaac.py


@dataclass
class ConvertCfg:
    min_extent_target: float = 0.065  # [m] cap on the smallest bbox extent
    max_extent_target: float = 0.22  # [m] cap on the largest bbox extent
    density: float = 350.0  # kg/m^3
    mass_min: float = 0.03
    mass_max: float = 0.5
    min_ok_extent: float = 0.012  # [m] thinner objects get ok=False
    n_points: int = 2048
    points_seed: int = 0
    static_friction: float = 1.0
    dynamic_friction: float = 0.9
    restitution: float = 0.0
    volume_grid: int = 64  # cells per axis for the hull-union volume
    flip_v: bool = False  # OBJ and USD share the uv origin, so no flip
    roughness: float = 0.6
    fmt: str = "usd"  # "usd" (crate, needs pxr) or "usda" (text)
    check_with_pxr: bool = True
    hull_mismatch_warn: float = 0.005  # [m] note if hull and visual bboxes differ by more


@dataclass
class ObjMesh:
    v: np.ndarray  # (N,3) float64
    f: np.ndarray  # (F,3) int64, 0-based
    vt: np.ndarray | None = None  # (M,2) float64
    ft: np.ndarray | None = None  # (F,3) int64, 0-based, into vt


def read_obj(path: str | os.PathLike) -> ObjMesh:
    """Read an OBJ file (v / vt / f); polygons are fan-triangulated and normals ignored."""
    vs, vts, fv, ft = [], [], [], []
    has_uv = True
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                vs.append(line.split()[1:4])
            elif line.startswith("vt "):
                vts.append(line.split()[1:3])
            elif line.startswith("f "):
                toks = line.split()[1:]
                iv, it = [], []
                for tok in toks:
                    parts = tok.split("/")
                    iv.append(int(parts[0]))
                    if len(parts) > 1 and parts[1]:
                        it.append(int(parts[1]))
                    else:
                        has_uv = False
                for k in range(1, len(iv) - 1):
                    fv.append((iv[0], iv[k], iv[k + 1]))
                    if has_uv:
                        ft.append((it[0], it[k], it[k + 1]))
    if not vs or not fv:
        raise ValueError(f"{path}: no vertices/faces")
    v = np.asarray(vs, dtype=np.float64)
    f = np.asarray(fv, dtype=np.int64)
    f = np.where(f < 0, f + len(v), f - 1)  # OBJ is 1-based, negative = relative
    if f.min() < 0 or f.max() >= len(v):
        raise ValueError(f"{path}: face index out of range")
    vt = t = None
    if has_uv and vts and ft:
        vt = np.asarray(vts, dtype=np.float64)
        t = np.asarray(ft, dtype=np.int64)
        t = np.where(t < 0, t + len(vt), t - 1)
        if t.min() < 0 or t.max() >= len(vt):
            raise ValueError(f"{path}: uv index out of range")
    return ObjMesh(v=v, f=f, vt=vt, ft=t)


def collision_paths(obj_dir: str | os.PathLike) -> list[str]:
    """Return the model_collision_*.obj paths sorted by integer index."""
    paths = glob.glob(os.path.join(str(obj_dir), "model_collision_*.obj"))
    return sorted(paths, key=lambda p: int(Path(p).stem.rsplit("_", 1)[1]))


def compute_scale(extents, cfg: ConvertCfg = ConvertCfg()) -> float:
    """Return s = min(1, 0.065 / e_min, 0.22 / e_max); it never enlarges."""
    e = np.sort(np.asarray(extents, dtype=np.float64))
    return float(min(1.0, cfg.min_extent_target / e[0], cfg.max_extent_target / e[-1]))


def vertex_normals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Return area-weighted vertex normals, merged across uv-seam duplicates to avoid shading seams."""
    _, inv = np.unique(np.round(v, 7), axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])  # |fn| = 2 * area
    acc = np.zeros((inv.max() + 1, 3))
    for k in range(3):
        np.add.at(acc, inv[f[:, k]], fn)
    n = acc[inv]
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.where(ln > 1e-20, n / np.maximum(ln, 1e-20), np.array([0.0, 0.0, 1.0]))
    return n


def hull_info(points: np.ndarray):
    """Return (facet equations (K,4), volume) of the convex hull; raises if degenerate."""
    from scipy.spatial import ConvexHull

    h = ConvexHull(points)
    return h.equations, float(h.volume)


def union_volume(hull_points: list[np.ndarray], hull_eqs: list[np.ndarray], n: int = 64) -> float:
    """Volume of the union of convex hulls on an n^3 grid over the hulls' joint bbox."""
    allp = np.concatenate(hull_points, axis=0)
    lo, hi = allp.min(0), allp.max(0)
    size = np.maximum(hi - lo, 1e-9)
    axes = [lo[i] + (np.arange(n) + 0.5) * size[i] / n for i in range(3)]
    P = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    inside = np.zeros(len(P), dtype=bool)
    for pts, eq in zip(hull_points, hull_eqs):
        hlo, hhi = pts.min(0), pts.max(0)
        m = ~inside & np.all(P >= hlo, axis=1) & np.all(P <= hhi, axis=1)
        idx = np.nonzero(m)[0]
        if len(idx) == 0:
            continue
        d = P[idx] @ eq[:, :3].T + eq[:, 3]
        inside[idx[d.max(axis=1) <= 1e-12]] = True
    return float(inside.mean() * np.prod(size))


def _f(x: float) -> str:
    return f"{x:.7g}"


def _vec3_list(a: np.ndarray) -> str:
    return ", ".join(f"({x:.7g}, {y:.7g}, {z:.7g})" for x, y, z in a.tolist())


def _vec2_list(a: np.ndarray) -> str:
    return ", ".join(f"({x:.7g}, {y:.7g})" for x, y in a.tolist())


def _int_list(a: np.ndarray) -> str:
    return ", ".join(map(str, np.asarray(a).reshape(-1).tolist()))


def _extent(p: np.ndarray) -> str:
    lo, hi = p.min(0), p.max(0)
    return f"[({_f(lo[0])}, {_f(lo[1])}, {_f(lo[2])}), ({_f(hi[0])}, {_f(hi[1])}, {_f(hi[2])})]"


def _usd_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def build_usda(
    name: str,
    vis_v: np.ndarray,
    vis_f: np.ndarray,
    vis_n: np.ndarray | None,
    uv: np.ndarray | None,
    uv_idx: np.ndarray | None,
    texture_path: str | None,
    hulls: list[tuple[np.ndarray, np.ndarray]],
    mass: float,
    meta: dict,
    cfg: ConvertCfg,
) -> str:
    """Return the USDA text of one object: /Object (rigid body) with /Looks, /visual and /collisions."""
    L: list[str] = []
    w = L.append
    w("#usda 1.0")
    w("(")
    w('    defaultPrim = "Object"')
    w("    metersPerUnit = 1")
    w('    upAxis = "Z"')
    w(f"    doc = {_usd_str(f'GSO object {name} converted by safegrasp.assets.gso')}")
    w("    customLayerData = {")
    for k, v in meta.items():
        w(f"        string {k} = {_usd_str(str(v))}")
    w("    }")
    w(")")
    w("")
    w('def Xform "Object" (')
    w('    kind = "component"')
    w('    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]')
    w(")")
    w("{")
    w(f"    float physics:mass = {_f(mass)}")
    w("    bool physics:rigidBodyEnabled = 1")
    w("")
    w('    def Scope "Looks"')
    w("    {")
    M = "/Object/Looks/VisualMaterial"
    w('        def Material "VisualMaterial"')
    w("        {")
    w(f"            token outputs:surface.connect = <{M}/PreviewSurface.outputs:surface>")
    w('            def Shader "PreviewSurface"')
    w("            {")
    w('                uniform token info:id = "UsdPreviewSurface"')
    if texture_path is not None:
        w(f"                color3f inputs:diffuseColor.connect = <{M}/DiffuseTexture.outputs:rgb>")
    else:
        w("                color3f inputs:diffuseColor = (0.6, 0.6, 0.6)")
    w(f"                float inputs:roughness = {_f(cfg.roughness)}")
    w("                float inputs:metallic = 0")
    w("                token outputs:surface")
    w("            }")
    if texture_path is not None:
        w('            def Shader "DiffuseTexture"')
        w("            {")
        w('                uniform token info:id = "UsdUVTexture"')
        w(f"                asset inputs:file = @{texture_path}@")
        w(f"                float2 inputs:st.connect = <{M}/STReader.outputs:result>")
        w('                token inputs:sourceColorSpace = "sRGB"')
        w('                token inputs:wrapS = "repeat"')
        w('                token inputs:wrapT = "repeat"')
        w("                float3 outputs:rgb")
        w("            }")
        w('            def Shader "STReader"')
        w("            {")
        w('                uniform token info:id = "UsdPrimvarReader_float2"')
        w('                string inputs:varname = "st"')
        w("                float2 outputs:result")
        w("            }")
    w("        }")
    w('        def Material "PhysicsMaterial" (')
    w('            prepend apiSchemas = ["PhysicsMaterialAPI"]')
    w("        )")
    w("        {")
    w(f"            float physics:staticFriction = {_f(cfg.static_friction)}")
    w(f"            float physics:dynamicFriction = {_f(cfg.dynamic_friction)}")
    w(f"            float physics:restitution = {_f(cfg.restitution)}")
    w("        }")
    w("    }")
    w("")
    w('    def Mesh "visual" (')
    w('        prepend apiSchemas = ["MaterialBindingAPI"]')
    w("    )")
    w("    {")
    w("        uniform bool doubleSided = 1")
    w(f"        float3[] extent = {_extent(vis_v)}")
    w(f"        int[] faceVertexCounts = [{_int_list(np.full(len(vis_f), 3))}]")
    w(f"        int[] faceVertexIndices = [{_int_list(vis_f)}]")
    w(f"        rel material:binding = <{M}>")
    if vis_n is not None:
        w(f"        normal3f[] normals = [{_vec3_list(vis_n)}] (")
        w('            interpolation = "vertex"')
        w("        )")
    w(f"        point3f[] points = [{_vec3_list(vis_v)}]")
    if uv is not None:
        w(f"        texCoord2f[] primvars:st = [{_vec2_list(uv)}] (")
        w('            interpolation = "faceVarying"')
        w("        )")
        w(f"        int[] primvars:st:indices = [{_int_list(uv_idx)}]")
    w('        uniform token subdivisionScheme = "none"')
    w('        uniform token purpose = "default"')
    w("    }")
    w("")
    # purpose "guide" keeps the collision hulls out of renders and camera images
    w('    def Xform "collisions"')
    w("    {")
    for i, (hv, hf) in enumerate(hulls):
        w(f'        def Mesh "c_{i}" (')
        w('            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "MaterialBindingAPI"]')
        w("        )")
        w("        {")
        w("            bool physics:collisionEnabled = 1")
        w('            uniform token physics:approximation = "convexHull"')
        w('            uniform token purpose = "guide"')
        w("            rel material:binding:physics = </Object/Looks/PhysicsMaterial> (")
        w('                bindMaterialAs = "weakerThanDescendants"')
        w("            )")
        w(f"            float3[] extent = {_extent(hv)}")
        w(f"            int[] faceVertexCounts = [{_int_list(np.full(len(hf), 3))}]")
        w(f"            int[] faceVertexIndices = [{_int_list(hf)}]")
        w(f"            point3f[] points = [{_vec3_list(hv)}]")
        w('            uniform token subdivisionScheme = "none"')
        w("        }")
    w("    }")
    w("}")
    w("")
    return "\n".join(L)


def pxr_env(isaacsim_root: str | None = None) -> dict:
    """Return env vars that let ``from pxr import Usd`` run in ~/isaacsim/python.sh without Kit."""
    root = isaacsim_root or os.path.expanduser("~/isaacsim")
    cands = sorted(glob.glob(os.path.join(root, "extscache", "omni.usd.libs-*")))
    if not cands:
        raise FileNotFoundError(f"omni.usd.libs not found under {root}/extscache")
    lib = cands[-1]
    env = {}
    env["PYTHONPATH"] = lib + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
    env["LD_LIBRARY_PATH"] = os.path.join(lib, "bin") + (
        ":" + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else ""
    )
    return env


def pxr_available() -> bool:
    try:
        from pxr import Sdf, Usd  # noqa: F401

        return True
    except Exception:
        return False


def write_usd_from_text(text: str, dst: str) -> None:
    """Write USDA text to ``dst``: ``.usda`` as text, ``.usd``/``.usdc`` as crate via pxr."""
    if dst.endswith(".usda"):
        with open(dst, "w") as fh:
            fh.write(text)
        return
    from pxr import Sdf

    layer = Sdf.Layer.CreateAnonymous(".usda")
    if not layer.ImportFromString(text):
        raise RuntimeError("pxr failed to parse generated usda")
    if not layer.Export(dst, args={"format": "usdc"}):
        raise RuntimeError(f"pxr failed to export {dst}")


def check_stage(path: str, n_hulls: int) -> list[str]:
    """Open the file with pxr, check the prim layout and return a list of problems."""
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    probs = []
    stage = Usd.Stage.Open(path)
    root = stage.GetDefaultPrim()
    if not root or root.GetPath() != "/Object":
        return ["defaultPrim is not /Object"]
    if not root.HasAPI(UsdPhysics.RigidBodyAPI):
        probs.append("no RigidBodyAPI")
    if not root.HasAPI(UsdPhysics.MassAPI):
        probs.append("no MassAPI")
    vis = stage.GetPrimAtPath("/Object/visual")
    if not vis or vis.HasAPI(UsdPhysics.CollisionAPI):
        probs.append("visual missing or has collision")
    else:
        st = UsdGeom.PrimvarsAPI(vis).GetPrimvar("st")
        if not st or st.GetInterpolation() != "faceVarying":
            probs.append("visual st primvar missing")
        mat, _ = UsdShade.MaterialBindingAPI(vis).ComputeBoundMaterial()
        if not mat:
            probs.append("visual material not bound")
    n = 0
    for p in stage.Traverse():
        if p.HasAPI(UsdPhysics.CollisionAPI):
            n += 1
            if p.GetAttribute("physics:approximation").Get() != "convexHull":
                probs.append(f"{p.GetPath()} approx != convexHull")
            if UsdGeom.Imageable(p).GetPurposeAttr().Get() != "guide":
                probs.append(f"{p.GetPath()} purpose != guide")
            pm, _ = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial("physics")
            if not pm or not pm.GetPrim().HasAPI(UsdPhysics.MaterialAPI):
                probs.append(f"{p.GetPath()} physics material not bound")
    if n != n_hulls:
        probs.append(f"collision prim count {n} != {n_hulls}")
    return probs


def convert_object(
    name: str,
    raw_dir: str | os.PathLike = RAW_DIR,
    usd_dir: str | os.PathLike = USD_DIR,
    points_dir: str | os.PathLike = POINTS_DIR,
    cfg: ConvertCfg = ConvertCfg(),
) -> dict:
    """Convert one GSO object. Never raises: failures are returned as ``ok=False`` entries."""
    import trimesh

    obj_dir = Path(raw_dir) / name
    ext = ".usda" if cfg.fmt == "usda" else ".usd"
    usd_path = Path(usd_dir).resolve() / f"{name}{ext}"
    pts_path = Path(points_dir).resolve() / f"{name}.npy"
    entry = dict(
        name=name, usd=str(usd_path), points=str(pts_path), scale=None, extents=None,
        extents_sorted=None, extents_raw=None, mass=None, volume=None, volume_hull_sum=None,
        n_hulls=0, n_hulls_raw=0, z_rest=None, n_vertices=0, n_faces=0, texture=None,
        ok=False, convert_ok=False, notes=[],
    )  # fmt: skip
    notes = entry["notes"]
    try:
        vis = read_obj(obj_dir / "model.obj")
    except Exception as e:  # noqa: BLE001
        notes.append(f"visual mesh failed to load: {e}")
        return entry
    lo, hi = vis.v.min(0), vis.v.max(0)
    ext_raw = hi - lo
    s = compute_scale(ext_raw, cfg)
    center = 0.5 * (lo + hi) * s
    V = vis.v * s - center
    entry.update(
        scale=s, extents_raw=ext_raw.tolist(), n_vertices=int(len(V)), n_faces=int(len(vis.f))
    )
    ext_s = V.max(0) - V.min(0)
    entry["extents"] = ext_s.tolist()
    entry["extents_sorted"] = np.sort(ext_s).tolist()

    tex = obj_dir / "texture.png"
    tex_path = str(tex.resolve()) if tex.is_file() else None
    entry["texture"] = tex_path
    if tex_path is None:
        notes.append("no texture.png (grey material)")
    uv = uv_idx = None
    if vis.vt is not None:
        uv = vis.vt.copy()
        if cfg.flip_v:
            uv[:, 1] = 1.0 - uv[:, 1]
        uv_idx = vis.ft
    else:
        notes.append("visual mesh has no uvs")

    hulls, hull_pts, hull_eqs, vols = [], [], [], []
    cpaths = collision_paths(obj_dir)
    entry["n_hulls_raw"] = len(cpaths)
    for cp in cpaths:
        try:
            cm = read_obj(cp)
            hv = cm.v * s - center
            eq, vol = hull_info(hv)
            if vol <= 1e-12:
                raise ValueError("zero volume")
        except Exception as e:  # noqa: BLE001
            notes.append(f"dropped collision piece {Path(cp).name}: {str(e).splitlines()[0][:80]}")
            continue
        hulls.append((hv, cm.f))
        hull_pts.append(hv)
        hull_eqs.append(eq)
        vols.append(vol)
    entry["n_hulls"] = len(hulls)
    if len(hulls) < 1:
        notes.append("no usable collision piece")
        return entry

    vol_union = union_volume(hull_pts, hull_eqs, cfg.volume_grid)
    mass = float(np.clip(vol_union * cfg.density, cfg.mass_min, cfg.mass_max))
    allh = np.concatenate(hull_pts, 0)
    entry.update(volume=vol_union, volume_hull_sum=float(sum(vols)), mass=mass)
    entry["z_rest"] = float(-allh[:, 2].min())  # origin height when resting upright on z=0
    mism = float(np.max(np.abs(np.concatenate([allh.min(0) - V.min(0), allh.max(0) - V.max(0)]))))
    entry["hull_bbox_mismatch"] = mism
    if mism > cfg.hull_mismatch_warn:
        notes.append(f"hull-union bbox differs from visual bbox by {mism * 1000:.1f} mm")

    try:
        tm = trimesh.Trimesh(vertices=V, faces=vis.f, process=False)
        pts, _ = trimesh.sample.sample_surface(tm, cfg.n_points, seed=cfg.points_seed)
        os.makedirs(pts_path.parent, exist_ok=True)
        np.save(pts_path, pts.astype(np.float32))
    except Exception as e:  # noqa: BLE001
        notes.append(f"surface sampling failed: {e}")
        return entry

    try:
        normals = vertex_normals(V, vis.f)
        meta = dict(
            gso_name=name, scale=f"{s:.8g}", mass=f"{mass:.6g}", volume=f"{vol_union:.6g}",
            source=str(obj_dir.resolve()),
        )  # fmt: skip
        text = build_usda(name, V, vis.f, normals, uv, uv_idx, tex_path, hulls, mass, meta, cfg)
        os.makedirs(usd_path.parent, exist_ok=True)
        write_usd_from_text(text, str(usd_path))
        if cfg.check_with_pxr and pxr_available():
            probs = check_stage(str(usd_path), len(hulls))
            if probs:
                notes.extend(f"stage check: {p}" for p in probs)
                return entry
    except Exception as e:  # noqa: BLE001
        notes.append(f"usd write failed: {e}")
        return entry

    if ext_s.min() < cfg.min_ok_extent:
        notes.append(f"scaled min extent {ext_s.min() * 1000:.1f} mm < {cfg.min_ok_extent * 1000:.0f} mm")
        return entry
    entry["ok"] = entry["convert_ok"] = True
    return entry


@dataclass
class GSOEntry:
    name: str
    usd: str  # absolute path, defaultPrim /Object
    points: str  # (2048,3) float32 points, object frame
    scale: float
    extents: list  # (3,) scaled bbox extents, object frame [m]
    extents_sorted: list
    mass: float  # [kg]
    volume: float  # [m^3] union of hulls
    n_hulls: int
    z_rest: float  # [m] origin height when resting upright on the table
    ok: bool
    notes: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)  # other json row fields

    _pts_cache: np.ndarray | None = field(default=None, repr=False, compare=False)

    def load_points(self) -> np.ndarray:
        """Return the (2048,3) float32 surface points in the object frame (cached)."""
        if self._pts_cache is None:
            self._pts_cache = np.load(self.points)
        return self._pts_cache

    def spawn_cfg(self, **kwargs):
        """Return an ``isaaclab.sim.UsdFileCfg`` for this object."""
        import isaaclab.sim as sim_utils

        return sim_utils.UsdFileCfg(usd_path=self.usd, **kwargs)

    def rigid_object_cfg(self, prim_path: str, pos=(0.0, 0.0, None), rot=(1.0, 0.0, 0.0, 0.0), **spawn_kwargs):
        """Return a ``RigidObjectCfg`` at ``prim_path``; ``pos[2] = None`` rests it upright on z = 0."""
        from isaaclab.assets import RigidObjectCfg

        pos = (float(pos[0]), float(pos[1]), float(self.z_rest if pos[2] is None else pos[2]))
        return RigidObjectCfg(
            prim_path=prim_path,
            spawn=self.spawn_cfg(**spawn_kwargs),
            init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=tuple(rot)),
        )


_ENTRY_FIELDS = {"name", "usd", "points", "scale", "extents", "extents_sorted", "mass", "volume", "n_hulls", "z_rest", "ok", "notes"}


def _resolve(p: str, base: Path) -> str:
    return p if os.path.isabs(p) else str((base / p).resolve())


def load_catalog(path: str | os.PathLike = CATALOG_PATH, ok_only: bool = True) -> list[GSOEntry]:
    """Load ``data/gso_catalog.json`` as a list of :class:`GSOEntry`, sorted by name."""
    path = Path(path)
    rows = json.loads(path.read_text())
    base = path.parent.parent  # repo root (catalog lives in data/)
    out = []
    for r in rows:
        if ok_only and not r.get("ok", False):
            continue
        out.append(
            GSOEntry(
                name=r["name"], usd=_resolve(r["usd"], base), points=_resolve(r["points"], base),
                scale=r["scale"], extents=r["extents"], extents_sorted=r["extents_sorted"],
                mass=r["mass"], volume=r["volume"], n_hulls=r["n_hulls"], z_rest=r.get("z_rest"),
                ok=r["ok"], notes=list(r.get("notes", [])),
                extra={k: v for k, v in r.items() if k not in _ENTRY_FIELDS},
            )  # fmt: skip
        )
    return sorted(out, key=lambda e: e.name)


def catalog_by_name(path: str | os.PathLike = CATALOG_PATH, ok_only: bool = True) -> dict[str, GSOEntry]:
    return {e.name: e for e in load_catalog(path, ok_only)}


def apply_validation(rows: list[dict], validation: dict) -> list[dict]:
    """Merge ``data/gso_validation.json`` into catalog rows, setting ``ok = convert_ok and sim_ok``.

    Sim failure reasons go into ``notes`` with a "sim: " prefix. Safe to run twice."""
    for r in rows:
        r.setdefault("convert_ok", bool(r.get("ok", False)))
        v = validation.get(r["name"])
        r["notes"] = [n for n in r.get("notes", []) if not n.startswith("sim: ")]
        if v is None:
            r["ok"] = bool(r["convert_ok"])
            continue
        r["sim_validated"] = True
        r["sim_ok"] = bool(v.get("ok", True))
        for k in ("z_bottom", "disp_xy", "lin_speed", "ang_speed", "tilt_deg"):
            if k in v:
                r[f"sim_{k}"] = v[k]
        r["ok"] = bool(r["convert_ok"]) and r["sim_ok"]
        if not r["sim_ok"]:
            r["notes"].extend(f"sim: {n}" for n in v.get("notes", []))
    return rows


def save_catalog(rows: list[dict], path: str | os.PathLike = CATALOG_PATH) -> None:
    rows = sorted(rows, key=lambda r: r["name"])
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(rows, fh, indent=1)
    os.replace(tmp, path)


def entry_to_row(e: GSOEntry) -> dict:
    d = asdict(e)
    d.pop("_pts_cache", None)
    extra = d.pop("extra", {})
    d.update(extra)
    return d


__all__ = [
    "ConvertCfg", "ObjMesh", "read_obj", "compute_scale", "union_volume", "build_usda",
    "convert_object", "GSOEntry", "load_catalog", "catalog_by_name", "apply_validation",
    "save_catalog", "pxr_env", "pxr_available", "check_stage", "RAW_DIR", "USD_DIR",
    "POINTS_DIR", "CATALOG_PATH", "VALIDATION_PATH",
]  # fmt: skip
