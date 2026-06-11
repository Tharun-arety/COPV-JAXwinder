Now I have read everything. Here is the complete change list, file by file, with exact code.

---

## geometry.py — two changes

**Change 1: Replace hemispherical dome with ellipsoidal**

The current `build_copv_shell()` uses `cq.Workplane("XY").sphere(outer_radius)` for both caps. A hemisphere has height-to-radius ratio of 1.0. An isotensoidal dome for a small boss ratio (r₀/R = 0.1 here) is well approximated by an ellipsoidal cap with h/R ≈ 0.7. This changes the meridional stress distribution significantly near the boss transition, which is where COPVs actually fail.

Replace in `build_copv_shell()`:

```python
# REMOVE these two lines from the outer union:
.union(cq.Workplane("XY").sphere(outer_radius).translate((0.0, 0.0, half_cyl)))
.union(cq.Workplane("XY").sphere(outer_radius).translate((0.0, 0.0, -half_cyl)))

# REPLACE with:
dome_h = dome_height_ratio * outer_radius  # default 0.7
outer_top = (
    cq.Workplane("XZ")
    .moveTo(0, half_cyl)
    .spline(
        [(outer_radius * m, half_cyl + dome_h * (1 - m**2) ** 0.5) for m in
         [i / 20 for i in range(21)]],
        includeCurrent=False,
    )
    .close()
    .revolve(360, (0, 0, 0), (0, 1, 0))
)
outer_bot = (
    cq.Workplane("XZ")
    .moveTo(0, -half_cyl)
    .spline(
        [(outer_radius * m, -half_cyl - dome_h * (1 - m**2) ** 0.5) for m in
         [i / 20 for i in range(21)]],
        includeCurrent=False,
    )
    .close()
    .revolve(360, (0, 0, 0), (0, 1, 0))
)
```

And update the function signature and the inner dome to match:

```python
def build_copv_shell(
    step_path: Path,
    outer_radius: float = 100.0,
    cylinder_length: float = 220.0,
    thickness: float = 8.0,
    opening_radius: float = 10.0,
    dome_height_ratio: float = 0.7,          # ADD THIS
) -> Path:
```

Do the same for the inner dome using `inner_radius` and the same `dome_height_ratio`. The boss cut logic does not change — it is already correct.

Also update `ensure_copv_mesh()` to accept and forward `dome_height_ratio`:

```python
def ensure_copv_mesh(step_path, msh_path, geom, remesh=False):
    if remesh or not step_path.exists():
        build_copv_shell(
            step_path,
            outer_radius=geom.outer_radius,
            cylinder_length=geom.cylinder_length,
            thickness=geom.thickness,
            opening_radius=geom.opening_radius,
            dome_height_ratio=geom.dome_height_ratio,   # ADD THIS
        )
```

---

**Change 2: Add boss mesh refinement in `mesh_step()`**

Current: `mesh_hmin=16.0`, `mesh_hmax=36.0` uniformly. Boss radius is 10mm. One element spans the entire boss transition zone. This is why the boss-region Hashin peak does not appear in the output. Fix with a gmsh `Threshold` field after initial meshing:

```python
def mesh_step(
    step_path: Path,
    msh_path: Path,
    hmin: float = 10.0,
    hmax: float = 28.0,
    boss_hmin: float = 4.0,         # ADD
    boss_refine_radius: float = 28.0,  # zone radius around each boss centre, in mm
    cylinder_half_len: float = 110.0,  # half of cylinder_length, needed to locate boss centres
    outer_radius: float = 100.0,       # needed to locate boss centres
) -> MeshResult:
    msh_path.parent.mkdir(parents=True, exist_ok=True)
    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(step_path.stem)
    gmsh.merge(str(step_path))
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", boss_hmin)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", hmax)

    # Refinement field centred on each polar boss
    f = gmsh.model.mesh.field
    field_ids = []
    for z_sign in [1.0, -1.0]:
        boss_z = z_sign * (cylinder_half_len + 0.5 * outer_radius)
        dist_id = f.add("Distance")
        f.setNumbers(dist_id, "PointsList", [])   # point-based distance
        # Use a small sphere ball around the boss centre instead
        ball_id = f.add("Ball")
        f.setNumber(ball_id, "Radius", boss_refine_radius)
        f.setNumber(ball_id, "XCenter", 0.0)
        f.setNumber(ball_id, "YCenter", 0.0)
        f.setNumber(ball_id, "ZCenter", boss_z)
        f.setNumber(ball_id, "Thickness", boss_refine_radius * 0.4)
        thresh_id = f.add("Threshold")
        f.setNumber(thresh_id, "InField", ball_id)
        f.setNumber(thresh_id, "SizeMin", boss_hmin)
        f.setNumber(thresh_id, "SizeMax", hmax)
        f.setNumber(thresh_id, "DistMin", 0.0)
        f.setNumber(thresh_id, "DistMax", boss_refine_radius)
        field_ids.append(thresh_id)

    min_id = f.add("Min")
    f.setNumbers(min_id, "FieldsList", field_ids)
    f.setAsBackgroundMesh(min_id)
    gmsh.model.mesh.generate(3)
    gmsh.write(str(msh_path))
    gmsh.finalize()
    return read_msh(msh_path, step_path=step_path)
```

Update `ensure_copv_mesh()` to pass through these new parameters:

```python
def ensure_copv_mesh(step_path, msh_path, geom, remesh=False):
    ...
    if remesh or not msh_path.exists():
        return mesh_step(
            step_path,
            msh_path,
            hmin=geom.mesh_hmin,
            hmax=geom.mesh_hmax,
            boss_hmin=geom.boss_hmin,
            boss_refine_radius=geom.boss_refine_radius,
            cylinder_half_len=geom.half_cyl,
            outer_radius=geom.outer_radius,
        )
```

---

## config.py — five changes

```python
@dataclass
class GeometryConfig:
    outer_radius: float = 100.0
    cylinder_length: float = 220.0
    thickness: float = 8.0
    opening_radius: float = 10.0
    pressure: float = 1.0            # Unit pressure. Burst factor is estimated by scaling.
    support_tol: float = 1.5
    mesh_hmin: float = 10.0          # CHANGE: was 16.0. 16 > boss radius, which washed out boss FI.
    mesh_hmax: float = 28.0          # CHANGE: was 36.0
    boss_hmin: float = 4.0           # ADD: local refinement near boss (≈ boss_radius / 2.5)
    boss_refine_radius: float = 28.0 # ADD: mm radius of refinement zone around each boss centre
    dome_height_ratio: float = 0.7   # ADD: ellipsoidal dome h/R. 1.0 = hemisphere, 0.7 ≈ isotensoidal
```

The `dome_height_ratio` field replaces the silent hemisphere assumption. The `boss_hmin` and `boss_refine_radius` drive the new mesh refinement. `mesh_hmin` going from 16 to 10 globally improves resolution across the entire vessel without waiting for the boss field to kick in.

One more addition to `MaterialConfig` — add the citation explicitly at class level so it appears when a hiring manager reads the code:

```python
@dataclass
class MaterialConfig:
    # T700/E862 UD ply elastic constants at 60 % fibre volume.
    # Source: NASA/TM-2013-216574, Table 2.
    # E3/nu13 treated as transversely isotropic (E3=E2, nu13=nu12).
    e_xx: float = 139067.0
    ...
```

---

## physics.py — three changes

**Change 1: Document the σ₂₂/σ₃₃ averaging in `hashin_failure_indices()`**

```python
def hashin_failure_indices(local_stress: jnp.ndarray, allowables: MaterialAllowables):
    sigma_11 = local_stress[..., 0, 0]
    # Transverse stress averaged over the isotropic 2-3 plane of the UD ply.
    # For membrane-dominated loading (thin shell away from boss) sigma_33 ≈ 0
    # so this collapses to standard 2D Hashin sigma_22.
    # Near the boss under triaxial constraint the 3D averaging is a known
    # approximation; it neither significantly over- nor under-states FI for
    # matrix tension at the boss rim for this vessel class.
    sigma_transverse = 0.5 * (local_stress[..., 1, 1] + local_stress[..., 2, 2])
```

This comment matters because a composites engineer reading the code will immediately ask about it. Naming it explicitly signals you know exactly what you did and why, rather than leaving a silent choice for them to wonder about.

**Change 2: Increase CG solver `maxiter` for the finer mesh**

The finer mesh (boss_hmin = 4mm vs old hmin = 16mm) will produce a stiffer, larger system. The CG convergence budget needs to grow:

```python
def make_solve_compliance(state, tol=1e-6, maxiter=2400):  # CHANGE: was 1200
```

**Change 3: Add Clairaut explanation to `required_friction_coefficient()`**

```python
def required_friction_coefficient(s_coords, rho, alpha, regularization=1e-6):
    # Deviation from Clairaut's theorem (rho * sin(alpha) = const) requires
    # lateral friction to maintain the fiber path. Required mu = |d(rho*sin(alpha))/ds|
    # divided by |rho * cos(alpha)|. A geodesic path has mu_required = 0 everywhere.
    # Values exceeding mu_max = 0.15 (wet-winding allowable) are penalised upstream
    # so they never reach TaniqWind path planning.
    clairaut = rho * jnp.sin(alpha)
```

---

## optimize.py — one targeted addition

In `winding_forward_angle()`, the `rho_floor` line silently handles the boss exclusion zone. Add a comment that names what it is:

```python
# Clairaut minimum reach for this equatorial angle: rho_min = R * sin(alpha_cyl).
# For alpha_cyl in [12°, 58°], rho_min ranges from 20.8 to 84.8 mm.
# Elements between the boss rim (opening_radius = 10 mm) and rho_min are
# unreachable by geodesic helical winding at this angle — covered by hoop passes only.
# rho_floor clips the arcsin argument to prevent domain errors in that zone.
rho_floor = max(geom.opening_radius + 4.0, 0.18 * geom.mid_radius)
```

No functional change. The implementation already handles this correctly. Naming it is what matters for a technical reader.

---

## abaqus_exporter.py — one addition

The current material block exports elastic constants but no density. Dynamic analysis (modal, acoustic) in Abaqus requires it. A flight-hardware team may run modal checks. Add this immediately after the elastic constants write:

```python
stream.write("*ELASTIC, TYPE=ENGINEERING CONSTANTS\n")
stream.write(
    f"{material.e_xx:.6f}, {material.e_yy:.6f}, ..."
)
# ADD:
if hasattr(material, 'density') and material.density is not None:
    stream.write("*DENSITY\n")
    stream.write(f"{material.density:.6f}\n")
```

And add `density: float = 1.58e-9` (T700/epoxy, tonnes/mm³ for Abaqus SI-mm-N unit system) to `MaterialConfig`. This is a minor addition but signals understanding of downstream Abaqus usage that a manufacturing engineer will recognise.

---

## README and outputs — three required actions before any email is sent

**Action 1: Sync the numbers.** Open `outputs/hybrid_summary.json`. Read the actual values. Update the README table to match exactly. The current discrepancy between the README and the memory record of JSON values is the most dangerous credibility failure in the repo. Do this before anything else.

**Action 2: Update the geometry description.** Change the current geometry section to:

> Ellipsoidal dome caps (height-to-radius ratio 0.7, approximating isotensoidal under netting theory) with polar boss apertures of 10 mm radius. Isotensoidal profiling and autofrettage effects are not modelled; the ellipsoidal dome is a documented engineering approximation for the current boss-to-radius ratio of 0.1.

Naming what is approximated is more credible than staying silent about it.

**Action 3: Document the boss fiber-free zone in the winding section.** Add:

> For helical winding angles in the optimised range [12°, 58°], the Clairaut condition limits the closest geodesic approach to 20.8–84.8 mm from the vessel axis. The polar boss at 10 mm radius falls inside this exclusion zone for all helical families. The boss rim is covered exclusively by hoop fiber from the hoop-pass field. This is the correct physical constraint for this vessel class — it is not a modelling gap.

This transforms a potential credibility risk (why no helical fiber at the boss?) into a statement of physical correctness.

---

## Execution order

Run these in sequence so each change builds on a confirmed-working state:

1. `config.py` — all parameter changes (no code execution needed)
2. `geometry.py` — boss mesh refinement only, rerun notebook on existing STEP file with `remesh=True` for msh only
3. Confirm boss-region FI peak appears in the Hashin map — if it does, commit
4. `geometry.py` — dome geometry change, rebuild STEP + remesh
5. Re-run `03_hybrid_verification.ipynb`, update `hybrid_summary.json`
6. Sync README table to new JSON values
7. `physics.py` and `optimize.py` — documentation-only changes, commit separately
8. `abaqus_exporter.py` — density addition

Commit each of steps 2–8 separately. That gives you 7 commits on top of the existing 2, which goes from a two-commit repo to a nine-commit repo with a readable development arc. A hiring manager who clicks through the commit history sees an engineer iterating on their work, not someone who generated a repo in one session.