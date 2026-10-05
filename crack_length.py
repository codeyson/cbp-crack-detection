"""
crack_length.py - crack mask -> total crack length in mm -> grade

    mask -> keep block interior -> fill small holes, drop small blobs -> skeleton (1 px centre line)
         -> prune short side spurs -> length in px (straight step 1, diagonal sqrt2) -> x mm_per_px

    from crack_length import crack_length_mm, grade_from_length
    res = crack_length_mm(mask, mm_per_px=0.10, valid_region=block["interior"])
    grade = grade_from_length(res["length_mm"], thr_a_mm, thr_b_mm)

Self-test: python crack_length.py      Dependencies: numpy, scipy, scikit-image

KNOWN BIASES:
  * Step counting overestimates straight cracks by 0% (0/45/90 deg) up to ~8% (near 20/70 deg). Not a constant
    factor: calibrate against independently measured crack lengths.
  * Rough mask edges make side spurs that inflate length. Annotate thin masks; set spur_px ~1.5x the mask width.
  * Thick masks shorten each crack end by about half the mask width.
  * Broken (fragmented) predictions undercount length.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

_S8 = np.ones((3, 3), bool)            # 8-connectivity
_K8 = np.ones((3, 3), int)
_K8[1, 1] = 0                          # counts the 8 neighbours


def remove_small_blobs(mask, max_px):
    """Drop connected pieces of mask with fewer than max_px pixels."""
    lab, n = ndi.label(mask, structure=_S8)
    keep = np.bincount(lab.ravel()) >= max_px
    keep[0] = False
    return keep[lab]


def prune_spurs(sk, k):
    """Remove side branches shorter than ~k px. Isolated pieces shorter than ~2k px also disappear."""
    x = sk.copy()
    for _ in range(k):                                    # 1) eat the ends k times
        x &= ndi.convolve(x.astype(int), _K8, mode="constant") > 1
    grow = x & (ndi.convolve(x.astype(int), _K8, mode="constant") == 1)   # 2) regrow the surviving ends
    for _ in range(k):                                    #    along the original skeleton
        grow = ndi.binary_dilation(grow, structure=_S8) & sk
    return x | grow


def skeleton_length_px(sk):
    """Straight steps count 1, diagonal steps sqrt2. A diagonal already linked through an L-corner is skipped."""
    s = sk.astype(bool)
    h = s[:, :-1] & s[:, 1:]
    v = s[:-1, :] & s[1:, :]
    d1 = s[:-1, :-1] & s[1:, 1:] & ~s[:-1, 1:] & ~s[1:, :-1]
    d2 = s[:-1, 1:] & s[1:, :-1] & ~s[:-1, :-1] & ~s[1:, 1:]
    return float(h.sum() + v.sum() + np.sqrt(2) * (d1.sum() + d2.sum()))


def crack_length_mm(mask, mm_per_px, min_area_px=30, spur_px=8, valid_region=None):
    """
    mask         : 2-D array, truthy = crack (threshold the model's probability first)
    mm_per_px    : scale at the block's TOP-SURFACE plane (from calibration)
    min_area_px  : blobs AND holes smaller than this are removed (holes make skeleton loops that inflate length)
    spur_px      : side spurs shorter than this are pruned; 0 disables
    valid_region : block interior mask (block_detect's b["interior"]) to keep edges/background out
    """
    m = np.asarray(mask).astype(bool)
    if valid_region is not None:
        m &= np.asarray(valid_region).astype(bool)
    m = ~remove_small_blobs(~m, min_area_px)              # fill small holes
    m = remove_small_blobs(m, min_area_px)
    sk = prune_spurs(skeletonize(m), spur_px)
    px = skeleton_length_px(sk)
    return {"length_mm": px * mm_per_px, "length_px": px,
            "n_pieces": int(ndi.label(sk, structure=_S8)[1]), "skeleton": sk}


def grade_from_length(length_mm, thr_a_mm, thr_b_mm, damage=False, margin_mm=0.0):
    """Thresholds have NO defaults on purpose: they come from the team/standard. Exactly on a threshold goes to
    the lower grade (team decision). margin_mm > 0 returns "UNSURE" near a threshold (trigger rule still open)."""
    if damage:
        return "C"
    if margin_mm > 0 and min(abs(length_mm - thr_a_mm), abs(length_mm - thr_b_mm)) <= margin_mm:
        return "UNSURE"
    return "A" if length_mm <= thr_a_mm else "B" if length_mm <= thr_b_mm else "C"


def run_tests():
    from skimage.draw import line
    L = lambda m, **kw: crack_length_mm(m, 1.0, **kw)["length_px"]
    blank = lambda: np.zeros((400, 400), bool)
    near = lambda got, want, pct: abs(got - want) <= want * pct / 100

    m = blank(); m[50, 10:110] = True
    assert near(L(m), 99, 2), "horizontal line"
    assert near(crack_length_mm(m, 0.25)["length_mm"], 99 * 0.25, 2), "mm scaling"
    region = blank(); region[:, 10:60] = True
    assert near(L(m, valid_region=region), 49, 3), "valid_region clips"
    m2 = m.copy(); m2[200:203, 200:203] = True
    assert near(L(m2), 99, 2), "tiny blob ignored"

    d = blank(); rr, cc = line(0, 0, 99, 99); d[rr, cc] = True
    assert near(L(d), 99 * np.sqrt(2), 1), "diagonal line (naive pixel count is 100)"

    band = blank(); band[46:55, 20:220] = True
    assert near(L(band), 199, 6), "thick band"

    holed = band.copy()                                    # pinholes used to read 523-937 px instead of ~199
    rng = np.random.default_rng(0)
    holed[rng.integers(46, 55, 60), rng.integers(20, 220, 60)] = False
    assert near(L(holed), L(band), 2), f"pinholes inflate length: {L(holed):.0f} vs {L(band):.0f}"

    rough = band.copy()
    for c in np.random.default_rng(1).integers(25, 215, 15):
        rough[44:46, c:c + 2] = True; rough[55:57, c:c + 2] = True
    assert L(rough, spur_px=12) < L(rough, spur_px=0) and near(L(rough, spur_px=12), 199, 3), "spur pruning"

    g = [grade_from_length(5, 10, 30), grade_from_length(20, 10, 30), grade_from_length(50, 10, 30),
         grade_from_length(1, 10, 30, damage=True), grade_from_length(29, 10, 30, margin_mm=2)]
    assert g == ["A", "B", "C", "C", "UNSURE"], g
    print("selftest: ALL PASS")


if __name__ == "__main__":
    run_tests()
