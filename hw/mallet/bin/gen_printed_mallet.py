#!/usr/bin/env python3
"""Generate a 3D-printable mallet whose centre of mass sits at the cable plane.

The net cable pull acts at one height and the mallet's inertia at its centre
of mass; any offset d tips it at about a = g * r / d, whatever it weighs. So
the low striking ring is balanced by metal and epoxy in a tower ABOVE the springs.

The spring clevises copy mallet-mount(5): two 5 mm lugs, a 5 mm gap, 2 mm
vertical pin holes at radius 31.74, 10 mm wide, tip at 36.74, with the ends
rounded concentric with the pins.

A continuous 45-degree shell carries puck hits from the ring into the spring
mount and carries the lugs; eight vertical holes keep the underside open.

Frame: z = 0 is the table, +z up, x and y along the lugs.
Prints bottom-down, solid (100% infill), so the mass model holds.

    pip install manifold3d trimesh numpy
    python hw/mallet/bin/gen_printed_mallet.py [--cable-z MM]

It prints the ballast mass that puts the centre of mass on the spring plane.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import trimesh
from manifold3d import CrossSection, JoinType, Manifold

OUT = Path(__file__).resolve().parents[1]

SEGMENTS = 240
PETG_G_CM3 = 1.27

MALLET_R = 50.4        # striking radius, as shared/cdpr_geometry MALLET_RADIUS_MM
RING_WALL = 4.0
RING_H = 20.0

# Spring mount, from mallet-mount(5)
LUG_T = 5.0
LUG_GAP = 5.0
LUG_W = 10.0
LUG_TIP_R = 36.75
PIN_R = 31.74
PIN_D = 2.0
MOUNT_ID_R = 23.0
MOUNT_OD_R = 26.74
MARKER_R = 26.8
MARKER_D = 7.0
MARKER_DEPTH = 0.2

SHELL_T = 3.0          # true thickness of the 45-degree shells
SHELL_Z = SHELL_T * math.sqrt(2)
SHELL_DROP = 3.5       # how far the shell passes below the lug tips
VENT_D = 13.0
VENT_R = 36.5

# Ballast pocket: scrap metal cast in epoxy, filled to the top.
BALLAST_R = 12.0
BALLAST_H = 16.0
TOWER_R = 14.4
TOWER_FLOOR_ABOVE_SHELL = 2.0
LID_PLUG = 2.0
LID_T = 2.0
LID_CLEAR = 0.1


def section(points):
    """A CrossSection from a simple polygon of either winding."""
    area = sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1]))
    return CrossSection([points if area > 0 else points[::-1]])


def revolve(points):
    return Manifold.revolve(section(points), SEGMENTS)


def cylinder(radius, z0, z1, segments=SEGMENTS):
    return Manifold.cylinder(z1 - z0, radius, radius, segments).translate((0, 0, z0))


def box(x0, x1, y0, y1, z0, z1):
    return Manifold.cube((x1 - x0, y1 - y0, z1 - z0)).translate((x0, y0, z0))


def lug_plan():
    """Lug outline: a full round end, concentric with the pin hole."""
    return (CrossSection.square((PIN_R - 25.0, LUG_W)).translate((25.0, -LUG_W / 2))
            + CrossSection.circle(LUG_W / 2, 96).translate((PIN_R, 0)))


def lug(z0):
    plate = Manifold.extrude(lug_plan(), LUG_T).translate((0, 0, z0))
    return plate - cylinder(PIN_D / 2, z0 - 1, z0 + LUG_T + 1, 48).translate((PIN_R, 0, 0))


def build_body(cable_z):
    mount_lo = cable_z - LUG_GAP / 2 - LUG_T
    mount_hi = cable_z + LUG_GAP / 2 + LUG_T

    ring_in = MALLET_R - RING_WALL
    ring = revolve([(ring_in, 0), (MALLET_R - 0.5, 0), (MALLET_R, 0.5), (MALLET_R, RING_H - 1),
                    (MALLET_R - 1, RING_H), (ring_in + 1, RING_H), (ring_in, RING_H - 1)])

    # Shell from the ring to the mount. Its top is cut flat at the lower lug's
    # top face, which keeps the spring gap clear all the way round.
    upper = lambda r: mount_lo - SHELL_DROP + LUG_TIP_R - r
    lower = lambda r: upper(r) - SHELL_Z
    lug_top = mount_lo + LUG_T
    flat_r = LUG_TIP_R + mount_lo - SHELL_DROP - lug_top
    shell = revolve([(ring_in + 0.5, lower(ring_in + 0.5)), (MOUNT_OD_R - 0.5, lower(MOUNT_OD_R - 0.5)),
                     (MOUNT_OD_R - 0.5, lug_top), (flat_r, lug_top), (ring_in + 0.5, upper(ring_in + 0.5))])
    # Everything above the shell's underside, to trim the lug pads against.
    above = revolve([(15.0, lower(15.0)), (55.0, lower(55.0)), (55.0, 80.0), (15.0, 80.0)])

    # The chamfered underside prints without support.
    mount = revolve([(MOUNT_ID_R, mount_lo + MOUNT_OD_R - MOUNT_ID_R), (MOUNT_OD_R, mount_lo),
                     (MOUNT_OD_R, mount_hi), (MOUNT_ID_R, mount_hi)])
    parts = [ring, mount, shell]
    for a in (0, 90, 180, 270):
        lugs = lug(mount_lo) + lug(cable_z + LUG_GAP / 2)
        # Extends the lower lug straight down into the shell: no overhang.
        pad = Manifold.extrude(lug_plan(), mount_lo - 10.0).translate((0, 0, 10.0)) ^ above
        parts.append((lugs + pad).rotate((0, 0, a)))

    # Inverted shell up to the ballast tower: leaves the marker face clear.
    top_z = mount_hi + (MOUNT_ID_R + 0.2 - TOWER_R)
    cone = revolve([(MOUNT_ID_R + 0.2, mount_hi), (TOWER_R - 1, top_z + 1),
                    (TOWER_R - 1, top_z + 1 - SHELL_Z), (MOUNT_ID_R + 0.2, mount_hi - SHELL_Z)])
    parts.append(cone)

    # Tied to the shell: any higher and the tower's underside misses it.
    floor = top_z + TOWER_FLOOR_ABOVE_SHELL
    pocket_top = floor + BALLAST_H
    tower_top = pocket_top + LID_PLUG
    # A pointed underside, so the tower floor needs no bridge or support.
    tip = floor - 1.5 - TOWER_R
    tower = revolve([(0, tip), (TOWER_R, tip + TOWER_R), (TOWER_R, tower_top), (0, tower_top)])
    tower -= cylinder(BALLAST_R, floor, tower_top + 1)
    parts.append(tower)

    body = parts[0]
    for p in parts[1:]:
        body += p
    for a in (0, 90, 180, 270):
        body -= cylinder(MARKER_D / 2, mount_hi - MARKER_DEPTH, mount_hi + 1, 64).translate((MARKER_R, 0, 0)).rotate((0, 0, a))
        vent = cylinder(VENT_D / 2, -1, lug_top - 0.1, 96).translate((VENT_R, 0, 0)).rotate((0, 0, a + 22.5))
        body -= vent + vent.rotate((0, 0, 45))
        # Clear through everything below the lugs, so a pin can be pushed out.
        body -= cylinder(PIN_D / 2, -1, mount_hi + 1, 48).translate((PIN_R, 0, 0)).rotate((0, 0, a))
    return body, dict(mount_lo=mount_lo, mount_hi=mount_hi, floor=floor, tower_top=tower_top)


def build_lid(tower_top):
    plug = cylinder(BALLAST_R - LID_CLEAR, 0, LID_PLUG)
    flange = cylinder(TOWER_R, LID_PLUG, LID_PLUG + LID_T)
    recess = cylinder(MARKER_D / 2, LID_PLUG + LID_T - MARKER_DEPTH, LID_PLUG + LID_T + 1, 64)
    return ((plug + flange) - recess).translate((0, 0, tower_top - LID_PLUG))


def to_trimesh(man):
    m = man.to_mesh()
    return trimesh.Trimesh(vertices=np.asarray(m.vert_properties)[:, :3], faces=np.asarray(m.tri_verts), process=False)


def report(body, lid, info, cable_z):
    assert len(body.decompose()) == 1, "the body is in separate pieces"
    b, l = to_trimesh(body), to_trimesh(lid)
    assert b.is_watertight and l.is_watertight
    masses = [("body", b.volume / 1000 * PETG_G_CM3, b.center_mass[2]),
              ("lid", l.volume / 1000 * PETG_G_CM3, l.center_mass[2])]
    z_ballast = info["floor"] + BALLAST_H / 2
    ballast = -sum(m * (z - cable_z) for _, m, z in masses) / (z_ballast - cable_z)
    volume = math.pi * BALLAST_R ** 2 * BALLAST_H / 1000
    masses.append(("ballast", ballast, z_ballast))
    total = sum(m for _, m, _ in masses)
    print("part            g        z mm")
    for name, m, z in masses:
        print(f"{name:12s} {m:7.1f} {z:9.1f}")
    print(f"total        {total:7.1f} g; cable plane z = {cable_z:.1f}")
    print(f"ballast: {ballast:.1f} g in {volume:.1f} cm^3 ({ballast / volume:.1f} g/cm^3), filled to the top")
    print(f"each gram of ballast moves the centre of mass {(z_ballast - cable_z) / total:.2f} mm")
    print(f"side markers z = {info['mount_hi'] - MARKER_DEPTH:.2f}, centre marker z = {info['tower_top'] + LID_T - MARKER_DEPTH:.2f}")
    return b, l


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cable-z", type=float, default=30.0, help="height of the springs above the table, mm")
    args = ap.parse_args()
    body, info = build_body(args.cable_z)
    lid = build_lid(info["tower_top"])
    b, l = report(body, lid, info, args.cable_z)
    b.export(OUT / "printed_mallet_body.stl")
    # The lid prints flange-down.
    l2 = l.copy()
    l2.apply_transform(trimesh.transformations.rotation_matrix(math.pi, [1, 0, 0]))
    l2.apply_translation([0, 0, -l2.bounds[0][2]])
    l2.export(OUT / "printed_mallet_lid.stl")


if __name__ == "__main__":
    main()
