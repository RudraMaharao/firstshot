import pytest
from fastapi.testclient import TestClient

from service.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_reference_data(client):
    palette = client.get("/api/palette").json()
    assert len(palette) == 18
    assert {p["key"] for p in palette if p["status"] == "restricted"} == {"lead_chrome", "cadmium_red"}
    assert len(client.get("/api/targets").json()) == 24
    assert {x["key"] for x in client.get("/api/legacy").json()} >= {"safety_yellow", "cobalt_sky"}
    assert "training" in client.get("/api/model-card").json()


def test_match_a_measured_sample(client):
    r = client.post("/api/match", json={"target": {"kind": "patch", "value": "blue sky"}})
    assert r.status_code == 200
    body = r.json()
    shares = sum(p["share"] for p in body["recipe"]["pigments"])
    assert abs(shares - 100.0) < 0.1
    assert set(body["recipe"]["delta_e"]) == {"D65", "A", "FL11", "LED-B4"}
    assert body["search"]["sets_checked"] < body["search"]["sets_total"]


def test_match_a_screen_color(client):
    r = client.post("/api/match", json={"target": {"kind": "hex", "value": "#3a6ea5"}, "priority": "cost"})
    assert r.status_code == 200
    body = r.json()
    assert body["target"]["spectral"] is False
    assert set(body["recipe"]["delta_e"]) == {"D65"}


def test_bad_input_is_a_422(client):
    assert client.post("/api/match", json={"target": {"kind": "hex", "value": "#12"}}).status_code == 422
    assert client.post("/api/match", json={"target": {"kind": "patch", "value": "plaid"}}).status_code == 422
    assert client.post("/api/match", json={"target": {"kind": "patch", "value": "orange"},
                                           "priority": "cheapest"}).status_code == 422
    assert client.post("/api/reformulate", json={"legacy": "unobtainium"}).status_code == 422


def test_trials_and_reformulation(client):
    t = client.post("/api/trials", json={"target": {"kind": "patch", "value": "orange"}}).json()
    assert [run["model"] for run in t["runs"]] == ["physics", "hybrid", "tuned"]
    r = client.post("/api/reformulate", json={"legacy": "signal_red", "priority": "cost"}).json()
    assert "cadmium_red" not in {p["key"] for p in r["replacement"]["pigments"]}
    assert r["change"]["cost_pct"] < 0
