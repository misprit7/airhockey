#!/usr/bin/env python3
"""Mallet position from the blob stream, alongside the puck.

WHY NOT track_mallet.locate()
    That takes a full IMAGE. blobtrack streams coordinates precisely so that a
    1440x1080 frame at 200 Hz (311 MB/s) never has to reach Python. Asking for
    images back would give up the frame rate the tracker exists to provide.

HOW THE MALLET IS SEPARATED FROM THE PUCK
    The puck is found FIRST, by solving its four-corner marker square
    (puck_stream.find_puck), and its corners are removed. Whatever is left is
    the mallet. Doing it in that order matters now that the puck is the
    cluster: any "which blob has the most neighbours" rule would happily
    return the puck.

    Two mallets exist and they are marked differently:

      markers=1  A HAND-HELD mallet, one dot. The player's hand wraps the
                 grip and covers anything stuck to the sides, so the puck
                 carries the cluster and the mallet carries the lone dot.
      markers=3  The ROBOT mallet, a centre marker plus two at 26.5 mm
                 radius. Unchanged, and still what log_hardware.py wants.

    Neither rule depends on brightness, which matters because every marker on
    this table is the same tape and a rule based on intensity would fail the
    moment one moved away from the camera nadir.

HEIGHT MATTERS MORE THAN IT LOOKS
    Back-projecting a marker at the wrong height puts it wrong by roughly the
    height error times the radial offset from the camera nadir over the camera
    height -- zero underneath the lens, growing to millimetres at the edges,
    always radial. That is the same lever that put 4.2 mm of error into the
    optical anchor measurements, so measure the dot's height and pass it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
import cdpr_geometry as geom  # noqa: E402
from puck_markers import LINK_MM, find_puck  # noqa: E402
from puck_stream import PuckTracker  # noqa: E402

try:
    from track_mallet import ARM_Z_MM  # measured, not inferred
except Exception:                       # noqa: BLE001
    ARM_Z_MM = 33.0

# Robot mallet only: its markers sit ARM_MARKER_R apart, so the cluster spans
# roughly 2 * 26.5. Generous enough to survive one marker being marginal.
CLUSTER_MM = 90.0
MIN_CLUSTER = 2          # 2 of 3 markers is enough for a centroid


class MalletTracker:
    """Mallet centroid in table mm, from the same blobs the puck comes from.

    Shares PuckTracker's rejection of glare, fixed markers and off-table
    blobs, because those are properties of the SCENE and duplicating them
    would mean two places to update when the table changes.
    """

    def __init__(self, tracker: PuckTracker | None = None,
                 marker_z_mm: float | None = None, markers: int = 3):
        """`markers` is 1 for a hand-held mallet, 3 for the robot's.

        `marker_z_mm` is the HEIGHT of the marker(s) above the playing
        surface; the default follows `markers` -- 33 mm for the robot's arm
        markers, 67 mm for a dot on top of a mallet. Getting it wrong is a
        parallax error proportional to radial distance from the camera nadir,
        so measure your own mallet rather than trusting either default.
        """
        if markers not in (1, 3):
            raise ValueError(f"markers must be 1 or 3, got {markers}")
        self.t = tracker or PuckTracker()
        self.markers = markers
        self.z = (marker_z_mm if marker_z_mm is not None else
                  (ARM_Z_MM if markers == 3 else geom.MALLET_Z_MM))
        self._last: tuple[float, float] | None = None

    def update(self, blobs):
        """Return (x, y, n_markers) in table mm, or None if not found."""
        kept, world_puck = self.t.candidates(blobs)
        if len(kept) == 0:
            return None

        # Take the puck out of the running before looking for anything else.
        free = np.ones(len(kept), bool)
        got = find_puck(world_puck)
        if got is not None:
            free[got[2]] = False
        if not free.any():
            return None

        # Back-project at the MALLET's height, not the puck's. Same pixels, a
        # different plane, and the difference is the whole point of this file.
        world = self.t._to_table(kept[free][:, :2], self.z)
        found = (self._lone(world, kept[free][:, 2]) if self.markers == 1
                 else self._cluster(world))
        if found is None:
            return None
        c, n = found
        self._last = (float(c[0]), float(c[1]))
        return self._last[0], self._last[1], n

    def _lone(self, world, bright):
        """A hand-held mallet: the one blob with nothing near it."""
        d = np.linalg.norm(world[:, None, :] - world[None, :, :], axis=2)
        alone = np.flatnonzero((d <= LINK_MM).sum(axis=1) == 1)
        if len(alone) == 0:
            return None
        if len(alone) > 1:
            # Several strays survive: believe the one nearest last frame's
            # mallet, and on the first frame the biggest blob — the same
            # tie-break the puck tracker used when the puck was the lone dot.
            i = (max(alone, key=lambda j: bright[j]) if self._last is None
                 else min(alone,
                          key=lambda j: np.hypot(*(world[j] - self._last))))
        else:
            i = int(alone[0])
        return world[i], 1

    def _cluster(self, world):
        """The robot mallet: seed on the blob with the most neighbours.

        Using the median of all candidates instead breaks as soon as anything
        else is on the table, because the median sits between the two objects
        and belongs to neither.
        """
        if len(world) < MIN_CLUSTER:
            return None
        d = np.linalg.norm(world[:, None, :] - world[None, :, :], axis=2)
        near = (d < CLUSTER_MM).sum(axis=1)
        seed = int(np.argmax(near))
        if near[seed] < MIN_CLUSTER:
            return None
        members = world[d[seed] < CLUSTER_MM]
        return members.mean(axis=0), len(members)


def _selftest() -> int:
    """Synthetic blobs: a four-corner puck plus a mallet, well separated."""
    import math

    calls = {}

    class FakeTracker:
        def candidates(self, blobs):
            return blobs, blobs[:, :2].astype(float)

        def _to_table(self, px, z):
            calls["z"] = z
            return px.astype(float)

    def make(markers, z):
        m = MalletTracker.__new__(MalletTracker)
        m.t, m.markers, m.z, m._last = FakeTracker(), markers, z, None
        return m

    r = geom.PUCK_MARKER_R_MM
    a = 0.4 + np.arange(4) * (math.pi / 2)
    puck = np.stack([1500.0 + r * np.cos(a), 300.0 + r * np.sin(a)], 1)

    def blobs(pts):
        pts = np.asarray(pts, float)
        return np.hstack([pts, np.full((len(pts), 1), 20.0)])

    # 1. Hand mallet: one dot, and the puck's four corners must not win.
    hand = np.array([[600.0, 700.0]])
    m = make(1, geom.MALLET_Z_MM)
    got = m.update(blobs(np.vstack([puck, hand])))
    assert got is not None, "lone dot not found"
    assert abs(got[0] - 600.0) < 1e-6 and abs(got[1] - 700.0) < 1e-6, got
    assert got[2] == 1, got
    assert calls["z"] == geom.MALLET_Z_MM, calls["z"]

    # 2. Puck alone must NOT be reported as a mallet.
    assert make(1, geom.MALLET_Z_MM).update(blobs(puck)) is None

    # 3. Robot mallet: the cluster, with the puck present and excluded.
    ar = geom.ARM_MARKER_R_MM
    robot = np.array([[900.0, 700.0], [900.0 - ar, 700.0],
                      [900.0 + ar, 700.0]])
    m = make(3, ARM_Z_MM)
    got = m.update(blobs(np.vstack([puck, robot])))
    assert got is not None and got[2] == 3, got
    assert abs(got[0] - 900.0) < 1e-6 and abs(got[1] - 700.0) < 1e-6, got
    assert calls["z"] == ARM_Z_MM, calls["z"]

    print("selftest PASSED — lone dot and 3-cluster both found with the "
          "puck's four corners on the table")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    from puck_stream import BlobStream

    n_markers = 1 if "--hand" in sys.argv else 3
    tr = PuckTracker()
    mt = MalletTracker(tr, markers=n_markers)
    st = BlobStream()
    print(f"mallet ({n_markers} marker(s), z={mt.z:.0f} mm) + puck "
          f"— ctrl-C to stop\n")
    n = 0
    for seq, t, blobs_ in st:
        n += 1
        if n % 20:
            continue
        mal = mt.update(blobs_)
        pk = tr.update(t, blobs_)
        ms = f"({mal[0]:7.1f},{mal[1]:6.1f}) n={mal[2]}" if mal else "--"
        ps = (f"({pk[0]:7.1f},{pk[1]:6.1f}) {tr.n_markers}/4" if pk else "--")
        print(f"[{t:8.3f}] mallet {ms:24s} puck {ps}")
