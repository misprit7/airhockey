#!/usr/bin/env python3
"""Mallet position from the blob stream, alongside the puck.

WHY NOT track_mallet.locate()
    That takes a full IMAGE. blobtrack streams coordinates precisely so that a
    1440x1080 frame at 200 Hz (311 MB/s) never has to reach Python. Asking for
    images back would give up the frame rate the tracker exists to provide.

HOW THE MALLET IS SEPARATED FROM THE PUCK
    The same fact PuckTracker already relies on, read the other way: the
    mallet carries THREE retroreflectors in a tight cluster and the puck
    carries ONE. PuckTracker takes the blob with no near neighbours; this
    takes the cluster.

    Neither depends on brightness, which matters because both are the same
    tape and a rule based on intensity would fail the moment the mallet moved
    away from the camera nadir.

HEIGHT MATTERS MORE THAN IT LOOKS
    The mallet's side markers sit at ARM_Z_MM = 33.0 above the surface,
    measured; the puck marker at 8.0. Back-projecting the mallet at the puck's
    height would put it wrong by roughly the height error times the radial
    offset from the camera nadir over the camera height -- zero underneath the
    lens and growing to millimetres at the edges, always radially. That is the
    same lever that made the optical anchor measurements drift, so it is worth
    being exact about.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from puck_stream import PuckTracker  # noqa: E402

try:
    from track_mallet import ARM_Z_MM  # measured, not inferred
except Exception:                       # noqa: BLE001
    ARM_Z_MM = 33.0

# Mallet markers sit ATTACH-ish radius apart, so a cluster spans roughly
# 2 * ARM_MARKER_R. Generous enough to survive one marker being marginal,
# tight enough that the puck never joins the cluster.
CLUSTER_MM = 90.0
MIN_CLUSTER = 2          # 2 of 3 markers is enough for a centroid


class MalletTracker:
    """Mallet centroid in table mm, from the same blobs the puck comes from.

    Shares PuckTracker's rejection of glare, fixed markers and off-table
    blobs, because those are properties of the SCENE and duplicating them
    would mean two places to update when the table changes.
    """

    def __init__(self, tracker: PuckTracker | None = None):
        self.t = tracker or PuckTracker()

    def update(self, blobs):
        """Return (x, y, n_markers) in table mm, or None if not found."""
        kept, _world_puck = self.t.candidates(blobs)
        if len(kept) < MIN_CLUSTER:
            return None

        # Back-project at the ARM height, not the puck's. Same pixels, a
        # different plane, and the difference is the whole point of this file.
        world = self.t._to_table(kept[:, :2], ARM_Z_MM)

        # Seed on the blob with the most neighbours within CLUSTER_MM. Using
        # the median of all candidates instead breaks as soon as the puck is
        # on the table, because the median sits between puck and mallet and
        # belongs to neither.
        d = np.linalg.norm(world[:, None, :] - world[None, :, :], axis=2)
        near = (d < CLUSTER_MM).sum(axis=1)
        seed = int(np.argmax(near))
        if near[seed] < MIN_CLUSTER:
            return None

        members = world[d[seed] < CLUSTER_MM]
        c = members.mean(axis=0)
        return float(c[0]), float(c[1]), len(members)


def _selftest() -> int:
    """Synthetic blobs: a 3-marker cluster plus a lone puck, well separated."""
    import types

    calls = {}

    class FakeTracker:
        def candidates(self, blobs):
            return blobs, None

        def _to_table(self, px, z):
            calls["z"] = z
            return px.astype(float)

    m = MalletTracker.__new__(MalletTracker)
    m.t = FakeTracker()

    mallet = np.array([[1000.0, 500.0], [1026.5, 500.0], [1013.0, 526.5]])
    puck = np.array([[400.0, 200.0]])
    blobs = np.vstack([puck, mallet])
    blobs = np.hstack([blobs, np.full((len(blobs), 1), 20.0)])

    got = m.update(blobs)
    assert got is not None, "cluster not found"
    x, y, n = got
    assert n == 3, f"expected 3 markers, got {n}"
    assert abs(x - mallet[:, 0].mean()) < 1e-6, x
    assert abs(y - mallet[:, 1].mean()) < 1e-6, y
    assert calls["z"] == ARM_Z_MM, f"back-projected at {calls['z']}, not the arm height"

    # Puck alone must NOT be reported as a mallet.
    assert m.update(np.hstack([puck, [[20.0]]])) is None
    print(f"selftest PASSED (centroid {x:.1f},{y:.1f} from {n} markers, "
          f"z={calls['z']} mm)")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    from puck_stream import BlobStream

    tr = PuckTracker()
    mt = MalletTracker(tr)
    st = BlobStream()
    print("mallet + puck — ctrl-C to stop\n")
    n = 0
    for seq, t, blobs in st:
        n += 1
        if n % 20:
            continue
        mal = mt.update(blobs)
        pk = tr.update(t, blobs)
        ms = f"({mal[0]:7.1f},{mal[1]:6.1f}) n={mal[2]}" if mal else "--"
        ps = f"({pk[0]:7.1f},{pk[1]:6.1f})" if pk else "--"
        print(f"[{t:8.3f}] mallet {ms:24s} puck {ps}")
