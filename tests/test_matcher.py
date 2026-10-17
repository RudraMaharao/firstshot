import numpy as np
import pytest

from firstshot.engine import KEY_TO_INDEX, get_engine
from firstshot.matcher import MIN_PASS, Candidate, Matcher, Target
from firstshot.pigments import N_PIGMENTS, RESTRICTED, WHITE


@pytest.fixture(scope="module")
def engine():
    return get_engine()


def candidate(pigments, de, p, cost, carbon):
    return Candidate(tuple(pigments), np.zeros(N_PIGMENTS), {"D65": de}, p, 0.1, cost, carbon)


def test_choose_prefers_fewest_passing_pigments():
    c1 = candidate([2], 1.4, 0.2, 1.0, 1.0)            # one pigment, fails
    c2 = candidate([2, 5], 0.6, 0.9, 1.3, 2.0)         # two pigments, passes
    c3 = candidate([2, 5, 7], 0.1, 1.0, 1.1, 1.5)      # three pigments, closest
    chosen, viable = Matcher.choose([c1, c2, c3], "fewest")
    assert viable and chosen is c2
    assert Matcher.choose([c1, c2, c3], "closest")[0] is c3
    assert Matcher.choose([c1, c2, c3], "cost")[0] is c3     # 15% cheaper than c2, so cost wins
    assert Matcher.choose([c1], "fewest") == (c1, False)


def test_cost_priority_prefers_fewer_pigments_when_costs_are_close():
    a = candidate([2, 5], 0.5, 0.9, 1.30, 2.0)
    b = candidate([2, 5, 7], 0.2, 0.9, 1.29, 2.0)
    assert Matcher.choose([a, b], "cost")[0] is a


def test_search_space_never_contains_restricted_pigments(engine):
    target = engine.target("patch", "orange")
    for exterior in (False, True):
        sets = engine.matcher.candidate_sets(target, "exhaustive", 3, exterior_only=exterior)
        used = {i for s in sets for i in s}
        assert not used & set(RESTRICTED)
        if exterior:
            assert KEY_TO_INDEX["ultramarine"] not in used and KEY_TO_INDEX["arylide_yellow"] not in used


def test_shortlist_cuts_the_search_space(engine):
    target = engine.target("patch", "blue sky")
    ai = engine.matcher.candidate_sets(target, "ai", 3)
    full = engine.matcher.candidate_sets(target, "exhaustive", 3)
    assert len(ai) <= 25 < len(full)


def test_recovers_a_reachable_color_with_few_pigments(engine):
    model = engine.models["tuned"]
    conc = np.zeros(N_PIGMENTS)
    conc[KEY_TO_INDEX["phthalo_blue"]] = 0.02
    conc[KEY_TO_INDEX["oxide_yellow"]] = 0.06
    conc[WHITE] = 0.92
    target = Target.from_reflectance(model.predict(conc[None])[0], "known recipe")
    result = engine.matcher.match(target, model, mode="ai")
    assert result.viable
    assert result.chosen.de["D65"] < 0.5
    assert result.chosen.n_colorants <= 2
    assert result.chosen.pass_probability >= MIN_PASS


def test_reformulation_removes_the_legacy_pigment(engine):
    out = engine.reformulate("safety_yellow", priority="cost")
    keys = {p["key"] for p in out["replacement"]["pigments"]}
    assert "lead_chrome" not in keys
    assert not keys & {"cadmium_red"}
    assert out["viable"]
    assert out["replacement"]["delta_e"]["D65"] <= 1.0


def test_unreplaceable_color_is_reported_not_hidden(engine):
    out = engine.reformulate("cobalt_sky", priority="cost")
    assert "cobalt_blue" not in {p["key"] for p in out["replacement"]["pigments"]}
    assert out["viable"] is False
