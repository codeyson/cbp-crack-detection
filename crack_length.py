"""
crack_length.py  -  crack mask -> total crack length in mm -> grade

Pipeline:  mask -> (restrict to block interior) -> drop tiny blobs -> skeletonize
           -> prune short spurs -> (optional) bridge small gaps -> length (px) x mm_per_px

Dependencies: numpy, scipy, scikit-image   (all preinstalled on Colab)
Run the self-tests:   python crack_length.py

KNOWN BIASES (measured in the self-tests, not hypothetical):
  * Step counting (1 and sqrt2) overestimates straight cracks by 0% (0/45/90 deg) up to
    ~8% (near 20/70 deg); ~5% on average for random orientations. It is NOT a constant
    factor, so calibrate against independently measured crack lengths on training data.
  * Rough mask edges wiggle the skeleton and can inflate length a lot (a rough 9 px band
    read ~2x too long with no pruning). Keep masks thin and consistent (annotate thin),
    set spur_px to about 1.5x the mask width, and/or use smooth_px.
  * Thick masks shorten each crack end by roughly half the mask width.
  * smooth_px (median filter) erases structures thinner than ~smooth_px+1 px, so it can
    delete hairline cracks. Use it only on thick masks.

Usage in a notebook:
    from crack_length import crack_length_mm, grade_from_length
    res = crack_length_mm(mask, mm_per_px=0.10, valid_region=block_interior_mask)
    print(res["length_mm"])
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from skimage.draw import line as draw_line

_K8 = np.ones((3, 3), int)
_K8[1, 1] = 0
_S8 = np.ones((3, 3), bool)


def _neighbors(sk):
    """Number of 8-neighbours for each skeleton pixel (0 elsewhere)."""
    return ndi.convolve(sk.astype(int), _K8, mode="constant") * sk


def _endpoints(sk):
    return sk & (_neighbors(sk) == 1)


def remove_small_blobs(mask, min_area_px):
    lab, n = ndi.label(mask, structure=_S8)
    if n == 0:
        return mask
    keep = np.bincount(lab.ravel()) >= min_area_px
    keep[0] = False
    return keep[lab]


def prune_spurs(sk, k):
    """Remove side branches shorter than ~k px (morphological pruning).
    Side effect: isolated skeleton pieces shorter than ~2k px disappear."""
    if k <= 0:
        return sk
    x = sk.copy()
    for _ in range(k):                       # 1) eat endpoints k times
        x = x & ~(x & (_neighbors(x) <= 1))
    grow = _endpoints(x)                     # 2) regrow only the true branch ends,
    for _ in range(k):                       #    constrained to the original skeleton
        grow = ndi.binary_dilation(grow, structure=_S8) & sk
    return x | grow


def bridge_endpoints(sk, max_gap_px):
    """Join each endpoint to the nearest pixel of a *different* piece within max_gap_px."""
    if max_gap_px <= 0:
        return sk
    from scipy.spatial import cKDTree
    ends = np.argwhere(_endpoints(sk))
    if len(ends) == 0:
        return sk
    lab, _ = ndi.label(sk, structure=_S8)
    pts = np.argwhere(sk)
    tree = cKDTree(pts)
    out = sk.copy()
    for r, c in ends:
        best, best_d = None, 1e18
        for j in tree.query_ball_point((r, c), max_gap_px):
            rr, cc = pts[j]
            if lab[rr, cc] != lab[r, c]:
                d = np.hypot(rr - r, cc - c)
                if d < best_d:
                    best, best_d = (rr, cc), d
        if best is not None:
            lr, lc = draw_line(r, c, best[0], best[1])
            out[lr, lc] = True
    return out


def skeleton_length_px(sk):
    """Length of a 1-px-wide skeleton: straight steps = 1, diagonal steps = sqrt(2).
    A diagonal is skipped when the two pixels are already linked through a shared
    straight neighbour (an L-corner), so corners are not double counted."""
    s = sk.astype(bool)
    h = s[:, :-1] & s[:, 1:]
    v = s[:-1, :] & s[1:, :]
    d1 = s[:-1, :-1] & s[1:, 1:] & ~s[:-1, 1:] & ~s[1:, :-1]
    d2 = s[:-1, 1:] & s[1:, :-1] & ~s[:-1, :-1] & ~s[1:, 1:]
    return float(h.sum() + v.sum() + np.sqrt(2) * (d1.sum() + d2.sum()))


def crack_length_mm(mask, mm_per_px, min_area_px=30, spur_px=8, bridge_px=0,
                    valid_region=None, smooth_px=0):
    """
    mask         : 2-D array, truthy = crack (threshold the model's probability first)
    mm_per_px    : scale at the block's TOP-SURFACE plane (from calibration)
    min_area_px  : blobs smaller than this are dropped (depends on resolution)
    spur_px      : spurs shorter than this are pruned; 0 disables
    bridge_px    : max gap to bridge between broken pieces; 0 disables.
                   Tune on training folds only.
    smooth_px    : median-filter radius applied to the mask before skeletonizing (0 = off)
    valid_region : optional boolean mask of the block interior. Pass an ERODED block
                   mask to keep block edges and background out of the measurement.
    """
    m = np.asarray(mask).astype(bool)
    if valid_region is not None:
        m = m & np.asarray(valid_region).astype(bool)
    if smooth_px > 0:
        m = ndi.median_filter(m.astype(np.uint8), size=2 * smooth_px + 1).astype(bool)
    m = remove_small_blobs(m, min_area_px)
    sk = skeletonize(m)
    sk = prune_spurs(sk, spur_px)
    sk = bridge_endpoints(sk, bridge_px)
    px = skeleton_length_px(sk)
    _, n_pieces = ndi.label(sk, structure=_S8)
    return {"length_mm": px * mm_per_px, "length_px": px,
            "n_pieces": int(n_pieces), "skeleton": sk}


def grade_from_length(length_mm, thr_a_mm, thr_b_mm, damage=False, margin_mm=0.0):
    """
    Placeholder grading rule. Thresholds have NO defaults on purpose: they must come
    from the team/standard. Whether a value exactly on a threshold belongs to the
    lower or upper grade (<= vs <) is also a team decision (here: <= goes lower).
    margin_mm > 0 returns "UNSURE" when the length is within margin of a threshold
    (one possible trigger for the "I'm not sure" path; the real rule is still open).
    """
    if damage:
        return "C"
    if margin_mm > 0 and (abs(length_mm - thr_a_mm) <= margin_mm or
                          abs(length_mm - thr_b_mm) <= margin_mm):
        return "UNSURE"
    if length_mm <= thr_a_mm:
        return "A"
    if length_mm <= thr_b_mm:
        return "B"
    return "C"


# --------------------------------------------------------------------------- tests
def _disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def run_tests():
    def check(name, got, expected, tol_pct, hi_pct=None):
        lo, hi = (-tol_pct, tol_pct) if hi_pct is None else (-tol_pct, hi_pct)
        err = 100 * (got - expected) / expected
        ok = lo <= err <= hi
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: got {got:.2f}, expected {expected:.2f} ({err:+.1f}%, allowed {lo:+.0f}..{hi:+.0f}%)")
        return ok

    results = []
    blank = lambda: np.zeros((400, 400), bool)

    # 1. horizontal 1-px line, 100 px long -> 99 steps
    m = blank(); m[50, 10:110] = True
    results.append(check("horizontal line", crack_length_mm(m, 1.0)["length_px"], 99, 2))

    # 2. diagonal line: naive pixel counting would give 100, true is 99*sqrt(2)
    m = blank(); rr, cc = draw_line(0, 0, 99, 99); m[rr, cc] = True
    naive = m.sum()
    results.append(check("diagonal line (naive count would be %d)" % naive,
                         crack_length_mm(m, 1.0)["length_px"], 99 * np.sqrt(2), 1))

    # 3. thick horizontal band (9 px wide, 200 long): expect slight shortening at the ends
    m = blank(); m[46:55, 20:220] = True
    results.append(check("thick band 9x200", crack_length_mm(m, 1.0)["length_px"], 199, 6))

    # 4. quarter-circle arc r=100, thickness 5
    th = np.linspace(0, np.pi / 2, 4000)
    arc = blank(); arc[np.round(100 * np.sin(th)).astype(int) + 20, np.round(100 * np.cos(th)).astype(int) + 20] = True
    thick5 = ndi.binary_dilation(arc, _disk(2))
    # allowed band is asymmetric on purpose: step counting overestimates curves (see docstring)
    results.append(check("arc, 5 px thick (step-count bias allowed)", crack_length_mm(thick5, 1.0)["length_px"], np.pi / 2 * 100, 3, 9))

    # 4b. orientation bias of step counting on straight 1-px lines (true length 200 px)
    worst = 0.0
    for ang in (0, 15, 22.5, 30, 45, 60, 67.5, 75, 90):
        mm_ = np.zeros((400, 400), bool)
        r1, c1 = int(round(200 * np.sin(np.radians(ang)))), int(round(200 * np.cos(np.radians(ang))))
        rr_, cc_ = draw_line(0, 0, r1, c1); mm_[rr_ + 20, cc_ + 20] = True
        e = 100 * (crack_length_mm(mm_, 1.0, spur_px=0)["length_px"] - np.hypot(r1, c1)) / np.hypot(r1, c1)
        worst = max(worst, e)
    ok = worst <= 9
    print(f"[{'PASS' if ok else 'FAIL'}] worst orientation overestimate on straight lines: {worst:+.1f}% (limit +9%)")
    results.append(ok)

    # 5. thickness invariance: same arc drawn 3 px vs 15 px thick
    l_thin = crack_length_mm(ndi.binary_dilation(arc, _disk(1)), 1.0)["length_px"]
    l_thick = crack_length_mm(ndi.binary_dilation(arc, _disk(7)), 1.0)["length_px"]
    results.append(check("thin vs thick mask of same crack", l_thick, l_thin, 5))

    # 6. fragmented line: 3 gaps of 4 px in a 300 px line
    m = blank(); m[100, 10:310] = True
    for g in (80, 160, 240):
        m[100, g:g + 4] = False
    no_bridge = crack_length_mm(m, 1.0, bridge_px=0)["length_px"]
    bridged = crack_length_mm(m, 1.0, bridge_px=8)["length_px"]
    results.append(check("fragmented, bridged", bridged, 299, 2))
    ok = bridged - no_bridge >= 8
    print(f"[{'PASS' if ok else 'FAIL'}] gaps cost length without bridging: {no_bridge:.1f} -> {bridged:.1f}")
    results.append(ok)

    # 7. tiny blob is ignored
    m = blank(); m[50, 10:110] = True; m[200:203, 200:203] = True
    results.append(check("tiny blob ignored", crack_length_mm(m, 1.0)["length_px"], 99, 2))

    # 8. mm scaling
    m = blank(); m[50, 10:110] = True
    results.append(check("mm scaling (0.25 mm/px)", crack_length_mm(m, 0.25)["length_mm"], 99 * 0.25, 2))

    # 9. valid_region excludes everything outside the block interior
    m = blank(); m[50, 10:110] = True
    region = np.zeros_like(m); region[:, 10:60] = True
    results.append(check("valid_region clips edge/background", crack_length_mm(m, 1.0, valid_region=region)["length_px"], 49, 3))

    # 10. rough band edges: spurs and wiggles inflate length
    def rough_band(seed, n):
        rng = np.random.default_rng(seed)
        b = blank(); b[46:55, 20:220] = True
        for c in rng.integers(25, 215, n):
            b[44:46, c:c + 2] = True; b[55:57, c:c + 2] = True
        return b
    mild = rough_band(1, 15)
    l0 = crack_length_mm(mild, 1.0, spur_px=0)["length_px"]
    l12 = crack_length_mm(mild, 1.0, spur_px=12)["length_px"]
    print(f"      (mild roughness: unpruned {l0:.1f}, spur_px=12 -> {l12:.1f})")
    results.append(l12 <= l0 and check("mild rough band, spur_px=12", l12, 199, 3))

    harsh = rough_band(0, 40)
    lh0 = crack_length_mm(harsh, 1.0, spur_px=0)["length_px"]
    lh12 = crack_length_mm(harsh, 1.0, spur_px=12)["length_px"]
    lhs = crack_length_mm(harsh, 1.0, spur_px=12, smooth_px=2)["length_px"]
    print(f"      (harsh roughness: unpruned {lh0:.1f}, pruned {lh12:.1f}, pruned+smooth_px=2 {lhs:.1f})")
    results.append(check("harsh rough band, pruned + smoothed", lhs, 199, 8))

    # 11. grading rule
    g = [grade_from_length(5, 10, 30), grade_from_length(20, 10, 30),
         grade_from_length(50, 10, 30), grade_from_length(1, 10, 30, damage=True),
         grade_from_length(29, 10, 30, margin_mm=2)]
    ok = g == ["A", "B", "C", "C", "UNSURE"]
    print(f"[{'PASS' if ok else 'FAIL'}] grade rule: {g}")
    results.append(ok)

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return all(results)


if __name__ == "__main__":
    raise SystemExit(0 if run_tests() else 1)
