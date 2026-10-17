import numpy as np
import pytest

from firstshot.colorimetry import delta_e2000, hex_to_lab, lab_to_hex, lab_to_xyz, reflectance_to_lab, xyz_to_lab
from firstshot.spectral import ILLUMINANTS, tables

# Sharma, Wu and Dalal (2005), "The CIEDE2000 color-difference formula: implementation notes,
# supplementary test data, and mathematical observations", Table 1 (selected pairs).
SHARMA = [
    ((50.0000, 2.6772, -79.7751), (50.0000, 0.0000, -82.7485), 2.0425),
    ((50.0000, 3.1571, -77.2803), (50.0000, 0.0000, -82.7485), 2.8615),
    ((50.0000, 2.8361, -74.0200), (50.0000, 0.0000, -82.7485), 3.4412),
    ((50.0000, 0.0000, 0.0000), (50.0000, -1.0000, 2.0000), 2.3669),
    ((50.0000, 2.5000, 0.0000), (73.0000, 25.0000, -18.0000), 27.1492),
    ((50.0000, 2.5000, 0.0000), (61.0000, -5.0000, 29.0000), 22.8977),
    ((50.0000, 2.5000, 0.0000), (56.0000, -27.0000, -3.0000), 31.9030),
    ((50.0000, 2.5000, 0.0000), (58.0000, 24.0000, 15.0000), 19.4535),
    ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
    ((63.0109, -31.0961, -5.8663), (62.8187, -29.7946, -4.0864), 1.2630),
    ((61.2901, 3.7196, -5.3901), (61.4292, 2.2480, -4.9620), 1.8731),
    ((35.0831, -44.1164, 3.7933), (35.0232, -40.0716, 1.5901), 1.8645),
    ((22.7233, 20.0904, -46.6940), (23.0331, 14.9730, -42.5619), 2.0373),
    ((36.4612, 47.8580, 18.3852), (36.2715, 50.5065, 21.2231), 1.4146),
    ((90.8027, -2.0831, 1.4410), (91.1528, -1.6435, 0.0447), 1.4441),
    ((90.9257, -0.5406, -0.9208), (88.6381, -0.8985, -0.7239), 1.5381),
    ((6.7747, -0.2908, -2.4247), (5.8714, -0.0985, -2.2286), 0.6377),
    ((2.0776, 0.0795, -1.1350), (0.9033, -0.0636, -0.5514), 0.9082),
]


def test_ciede2000_matches_published_test_data():
    lab1 = np.array([p[0] for p in SHARMA])
    lab2 = np.array([p[1] for p in SHARMA])
    expected = np.array([p[2] for p in SHARMA])
    np.testing.assert_allclose(delta_e2000(lab1, lab2), expected, atol=1e-4)


def test_ciede2000_is_symmetric_and_zero_for_identical_colors():
    rng = np.random.default_rng(0)
    a = rng.uniform([0, -80, -80], [100, 80, 80], (200, 3))
    b = a + rng.normal(0, 4, a.shape)
    np.testing.assert_allclose(delta_e2000(a, b), delta_e2000(b, a), atol=1e-9)
    np.testing.assert_allclose(delta_e2000(a, a), 0.0, atol=1e-9)


def test_lab_roundtrip():
    white = tables().white["D65"]
    lab = np.array([[52.0, 10.5, -30.2], [5.0, 0.3, -0.2], [96.0, -1.0, 3.0]])
    np.testing.assert_allclose(xyz_to_lab(lab_to_xyz(lab, white), white), lab, atol=1e-9)


def test_perfect_white_is_L100_under_every_light():
    for name in ILLUMINANTS:
        np.testing.assert_allclose(reflectance_to_lab(np.ones(31), name), [100.0, 0.0, 0.0], atol=1e-9)


def test_hex_roundtrip():
    for hex_value in ("#3a6ea5", "#c9572f", "#f5f5ef", "#1b1b1a"):
        assert lab_to_hex(hex_to_lab(hex_value)) == hex_value


def test_matches_colour_science_exactly():
    colour = pytest.importorskip("colour")
    t = tables()
    shape = colour.SpectralShape(400, 700, 10)
    cmfs = colour.MSDS_CMFS["CIE 1964 10 Degree Standard Observer"].copy().align(shape)
    for name in ILLUMINANTS:
        illuminant = colour.SDS_ILLUMINANTS[name].copy().align(shape)
        ones = colour.SpectralDistribution(dict(zip(t.wavelengths, np.ones(31))))
        white_xy = colour.XYZ_to_xy(colour.sd_to_XYZ(ones, cmfs, illuminant, method="Integration") / 100)
        for refl in t.patch_reflectance[::3]:
            sd = colour.SpectralDistribution(dict(zip(t.wavelengths, refl)))
            xyz = colour.sd_to_XYZ(sd, cmfs, illuminant, method="Integration")
            np.testing.assert_allclose(reflectance_to_lab(refl, name), colour.XYZ_to_Lab(xyz / 100, white_xy),
                                       atol=1e-8)
    rng = np.random.default_rng(1)
    a = rng.uniform([0, -80, -80], [100, 80, 80], (300, 3))
    b = a + rng.normal(0, 5, a.shape)
    np.testing.assert_allclose(delta_e2000(a, b), colour.delta_E(a, b, method="CIE 2000"), atol=1e-9)
