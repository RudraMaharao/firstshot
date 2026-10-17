import numpy as np

from firstshot import km
from firstshot.colorimetry import reflectance_to_lab
from firstshot.pigments import COLORANTS, EXTERIOR, K_matrix, N_PIGMENTS, RESTRICTED, S_matrix, WHITE, palette


def tint(key: str, share: float) -> np.ndarray:
    i = [p.key for p in palette()].index(key)
    c = np.zeros(N_PIGMENTS)
    c[i] = share
    c[WHITE] += 1.0 - share
    return km.reflectance(c, K_matrix(), S_matrix())[0]


def test_r_infinity_limits_and_monotonic():
    q = np.array([1e-9, 0.01, 0.1, 1.0, 10.0])
    r = km.r_infinity(q, np.ones_like(q))
    assert abs(r[0] - 1.0) < 1e-3
    assert np.all(np.diff(r) < 0)
    assert np.all((r > 0) & (r <= 1))


def test_saunderson_keeps_reflectance_physical():
    r = np.linspace(0.0, 1.0, 11)
    m = km.saunderson(r)
    assert np.all(np.diff(m) > 0)
    assert abs(m[0] - km.K1) < 1e-12
    assert np.all(m <= 1.0)


def test_titanium_white_is_near_neutral_and_bright():
    lab = reflectance_to_lab(tint("white", 1.0))
    assert lab[0] > 95 and abs(lab[1]) < 2 and abs(lab[2]) < 3


def test_more_colorant_means_darker_tint():
    lightness = [reflectance_to_lab(tint("phthalo_blue", s))[0] for s in (0.01, 0.05, 0.2)]
    assert lightness[0] > lightness[1] > lightness[2]


def test_pigment_tints_land_in_the_right_hue():
    def lab(key):
        return reflectance_to_lab(tint(key, 0.10))
    for key in ("phthalo_blue", "ultramarine", "cobalt_blue"):
        assert lab(key)[2] < -25, key
    for key in ("arylide_yellow", "bismuth_yellow", "isoindoline_yellow", "lead_chrome"):
        assert lab(key)[2] > 45, key
    for key in ("pyrrole_red", "cadmium_red", "quin_magenta"):
        assert lab(key)[1] > 30, key
    assert lab("phthalo_green")[1] < -30


def test_restricted_pigments_are_never_colorants():
    assert set(RESTRICTED) == {i for i, p in enumerate(palette()) if p.status == "restricted"}
    assert not set(RESTRICTED) & set(COLORANTS)
    assert set(EXTERIOR) <= set(COLORANTS)
    names = {palette()[i].key for i in COLORANTS} - {palette()[i].key for i in EXTERIOR}
    assert names == {"ultramarine", "arylide_yellow"}
