import datetime
import json
import os
import socket
import threading
import time
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from src import config, database, preprocessing, retrain
from src.prediction import store

app = FastAPI(title="ECG Classification API", version="2.0",
              description="Classifies 12-lead ECG images into Normal, Abnormal_Heartbeat, MI and History_of_MI.")
database.init_db()

START_TIME = time.time()
STARTED_AT = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
HOSTNAME = socket.gethostname()

_features_lock = threading.Lock()
_features_cache = {"key": None, "rows": []}


def _classes():
    if os.path.exists(config.CLASS_NAMES_PATH):
        with open(config.CLASS_NAMES_PATH) as f:
            return json.load(f)
    if os.path.isdir(config.TRAIN_DIR):
        return sorted(d for d in os.listdir(config.TRAIN_DIR)
                      if os.path.isdir(os.path.join(config.TRAIN_DIR, d)))
    return []


def _count_images(folder):
    counts = {}
    if not os.path.isdir(folder):
        return counts
    for cls in sorted(os.listdir(folder)):
        path = os.path.join(folder, cls)
        if os.path.isdir(path):
            counts[cls] = len([f for f in os.listdir(path) if f.lower().endswith(preprocessing.ALLOWED_EXT)])
    return counts


@app.get("/health")
def health():
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "started_at": STARTED_AT,
        "served_by": HOSTNAME,
        "model_loaded": os.path.exists(config.MODEL_PATH),
        "model_last_updated": store.model_version(),
        "classes": _classes(),
    }


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    started = time.time()
    data = file.file.read()
    if not preprocessing.is_valid_image(data):
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")
    try:
        result = store.predict_bytes(data)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    result["latency_ms"] = round((time.time() - started) * 1000, 1)
    result["served_by"] = HOSTNAME
    result["model_last_updated"] = store.model_version()
    return result


@app.post("/upload")
def upload(label: str = Form(...), files: List[UploadFile] = File(...)):
    classes = _classes()
    if label not in classes:
        raise HTTPException(status_code=400, detail=f"Unknown label '{label}'. Choose one of {classes}.")
    saved, rejected = 0, []
    for f in files:
        data = f.file.read()
        if not (f.filename or "").lower().endswith(preprocessing.ALLOWED_EXT) or not preprocessing.is_valid_image(data):
            rejected.append(f.filename)
            continue
        path = preprocessing.save_upload(label, f.filename, data)
        database.add_upload(os.path.basename(path), label, path)
        saved += 1

    pending = database.count_pending()
    auto_started = False
    if pending >= config.RETRAIN_THRESHOLD:
        auto_started = retrain.start_retraining_async()
    return {
        "saved": saved,
        "rejected": rejected,
        "pending_uploads": pending,
        "retrain_threshold": config.RETRAIN_THRESHOLD,
        "auto_retrain_started": auto_started,
    }


@app.post("/retrain")
def trigger_retrain(epochs: int = 40):
    # Retraining with no new uploads is allowed: it continues training the current model on data/train
    started = retrain.start_retraining_async(epochs=max(1, min(epochs, 100)))
    if not started:
        raise HTTPException(status_code=409, detail="A retraining run is already in progress.")
    return {"started": True, "pending_uploads": database.count_pending()}


@app.get("/retrain/status")
def retrain_status():
    return retrain.get_status()


@app.get("/stats")
def stats():
    metrics = None
    if os.path.exists(config.METRICS_PATH):
        with open(config.METRICS_PATH) as f:
            metrics = json.load(f)
    return {
        "classes": _classes(),
        "train_counts": _count_images(config.TRAIN_DIR),
        "test_counts": _count_images(config.TEST_DIR),
        "pending_uploads": database.count_pending(),
        "pending_by_label": database.pending_by_label(),
        "total_uploads": database.total_uploads(),
        "retrain_threshold": config.RETRAIN_THRESHOLD,
        "metrics": metrics,
        "retrain_runs": database.list_runs(10),
    }


@app.get("/data/features")
def data_features():
    # Per-image features for the visualizations; cached because it opens every image once
    rows, key_parts = [], []
    for split_dir in (config.TRAIN_DIR, config.TEST_DIR):
        for cls, n in _count_images(split_dir).items():
            key_parts.append(f"{split_dir}:{cls}:{n}")
    key = "|".join(key_parts)
    with _features_lock:
        if _features_cache["key"] == key:
            return _features_cache["rows"]
        for split_dir in (config.TRAIN_DIR, config.TEST_DIR):
            if not os.path.isdir(split_dir):
                continue
            for cls in sorted(os.listdir(split_dir)):
                folder = os.path.join(split_dir, cls)
                if not os.path.isdir(folder):
                    continue
                for name in os.listdir(folder):
                    if name.lower().endswith(preprocessing.ALLOWED_EXT):
                        feats = preprocessing.describe_image(os.path.join(folder, name))
                        feats["label"] = cls
                        feats["split"] = os.path.basename(split_dir)
                        rows.append(feats)
        _features_cache.update(key=key, rows=rows)
    return rows


@app.get("/")
def root():
    return {"service": "ECG Classification API", "docs": "/docs", "health": "/health",
            "endpoints": ["/predict", "/upload", "/retrain", "/retrain/status", "/stats", "/data/features"]}
