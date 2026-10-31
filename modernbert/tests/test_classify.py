"""Quick smoke tests — run against a live container: pytest tests/ --base-url http://localhost:7998"""
import os

import pytest
import requests

BASE = os.environ.get("MODERNBERT_BASE_URL", "http://localhost:7998")


def test_health():
    r = requests.get(f"{BASE}/health")
    assert r.status_code == 200
    assert "model_state" in r.json()


def test_status():
    r = requests.get(f"{BASE}/model/status")
    assert r.status_code == 200
    d = r.json()
    assert d["model_id"] == "answerdotai/ModernBERT-base"


def test_update_labels_and_classify():
    labels = ["curious", "defensive", "playful", "sad", "angry", "flirtatious", "neutral"]
    r = requests.put(f"{BASE}/labels", json={"labels": labels})
    assert r.status_code == 200

    r = requests.post(f"{BASE}/classify", json={
        "text": "why won't you tell me the truth?",
        "context": ["I asked you about that before", "you always dodge the question"],
    })
    assert r.status_code == 200
    d = r.json()
    assert len(d["labels"]) > 0
    assert len(d["labels"]) == len(d["scores"])
    assert all(label in labels for label in d["labels"])


def test_embed():
    r = requests.post(f"{BASE}/embed", json={"texts": ["hello world", "test embedding"]})
    assert r.status_code == 200
    d = r.json()
    assert len(d["embeddings"]) == 2
    assert d["dim"] > 0


def test_evict_and_reload():
    requests.post(f"{BASE}/model/load")
    r = requests.get(f"{BASE}/model/status")
    assert r.json()["state"] in ("gpu", "cpu")

    requests.post(f"{BASE}/model/evict")
    r = requests.get(f"{BASE}/model/status")
    assert r.json()["state"] in ("cpu", "unloaded")

    requests.post(f"{BASE}/model/unload")
    r = requests.get(f"{BASE}/model/status")
    assert r.json()["state"] == "unloaded"
