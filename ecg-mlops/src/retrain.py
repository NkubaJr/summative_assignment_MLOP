"""Retraining pipeline and trigger.

Triggered by the UI button, POST /retrain, or automatically once RETRAIN_THRESHOLD new
labelled images are waiting. Steps: stage uploads into data/train -> preprocess (same
pipeline as training) -> load the CURRENT saved model as the pretrained starting point ->
continue training -> evaluate on the fixed test set -> archive the old model and publish
the new one if it is not worse.
"""
import datetime
import json
import os
import shutil
import threading

from src import config, database, preprocessing
from src.model import evaluate_model, load_model

# A retrained model replaces the current one unless its macro F1 on the test set drops by more than this
MAX_F1_DROP = 0.02

_state_lock = threading.Lock()
_state = {
    "running": False,
    "stage": "idle",
    "message": "No retraining has run yet.",
    "run_id": None,
    "started_at": None,
    "finished_at": None,
    "last_metrics": None,
}


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _set(**kwargs):
    with _state_lock:
        _state.update(kwargs)


def get_status():
    with _state_lock:
        status = dict(_state)
    if not status["running"]:
        active = database.running_run()
        if active:
            status.update(running=True, stage="training", run_id=active["id"],
                          message="Retraining is running in another container.")
    return status


def start_retraining_async(epochs=40, learning_rate=3e-4):
    """Start retraining in a background thread. Returns False if one is already running."""
    if database.running_run() is not None:
        return False
    with _state_lock:
        if _state["running"]:
            return False
        _state.update(running=True, stage="queued", message="Retraining queued.",
                      started_at=_now(), finished_at=None)
    threading.Thread(target=run_retraining, args=(epochs, learning_rate), daemon=True).start()
    return True


def run_retraining(epochs=40, learning_rate=3e-4):
    run_id = None
    try:
        # 1. Data: copy newly uploaded images into the training folders
        _set(running=True, stage="staging", message="Adding uploaded images to the training data.")
        pending = database.pending_uploads()
        run_id = database.start_run(len(pending))
        _set(run_id=run_id)
        for row in pending:
            if os.path.exists(row["path"]):
                dst_dir = os.path.join(config.TRAIN_DIR, row["label"])
                os.makedirs(dst_dir, exist_ok=True)
                shutil.copy(row["path"], os.path.join(dst_dir, os.path.basename(row["path"])))

        # 2. Preprocessing: same trace extraction + HOG pipeline as the original training
        _set(stage="preprocessing", message="Extracting features from training and test images.")
        train_paths, train_labels = preprocessing.list_images(config.TRAIN_DIR)
        test_paths, test_labels = preprocessing.list_images(config.TEST_DIR)
        prog = lambda i, n: _set(message=f"Extracting features: {i}/{n} images")
        X_train = preprocessing.features_from_paths(train_paths, config.FEATURE_CACHE_PATH, prog)
        X_test = preprocessing.features_from_paths(test_paths, config.FEATURE_CACHE_PATH)

        # 3. The current saved model is the pretrained starting point
        _set(stage="loading_model", message="Loading the current model as the pretrained model.")
        model = load_model(config.MODEL_PATH)
        unknown = sorted(set(train_labels) - set(model.class_names))
        if unknown:
            raise RuntimeError(f"Training folders contain classes the model does not know: {unknown}")
        before = evaluate_model(model, X_test, test_labels)

        # 4. Continue training
        _set(stage="training", message="Training started.")

        def log(rec):
            _set(message=(f"Epoch {rec['epoch']}: val_accuracy={rec['val_accuracy']:.3f} "
                          f"val_loss={rec['val_loss']:.3f}"))
        model.continue_training(X_train, train_labels, learning_rate=learning_rate,
                                max_epochs=epochs, log=log)

        # 5. Evaluate, then publish only if the model did not get worse
        _set(stage="evaluating", message="Evaluating on the held-out test set.")
        after = evaluate_model(model, X_test, test_labels)
        promoted = after["f1_macro"] >= before["f1_macro"] - MAX_F1_DROP
        notes = (f"accuracy {before['accuracy']:.3f} -> {after['accuracy']:.3f}, "
                 f"macro F1 {before['f1_macro']:.3f} -> {after['f1_macro']:.3f}, "
                 f"{len(pending)} new image(s), {len(train_paths)} training images")
        if promoted:
            os.makedirs(config.ARCHIVE_DIR, exist_ok=True)
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            shutil.copy(config.MODEL_PATH, os.path.join(config.ARCHIVE_DIR, f"ecg_hog_mlp_{stamp}.pkl"))
            tmp = config.MODEL_PATH + ".tmp"
            model.save(tmp)
            os.replace(tmp, config.MODEL_PATH)        # atomic swap: readers never see a half-written file
            saved = dict(after)
            saved["previous"] = {k: before[k] for k in ("accuracy", "f1_macro", "precision_macro", "recall_macro")}
            saved["updated_at"] = _now()
            saved["source"] = "retraining"
            with open(config.METRICS_PATH, "w") as f:
                json.dump(saved, f, indent=2)
            notes = "New model published. " + notes
        else:
            notes = "Kept the previous model (the retrained one scored worse). " + notes

        database.mark_uploads_used([row["id"] for row in pending])
        database.finish_run(run_id, "completed", after, notes)
        _set(running=False, stage="done", message="Retraining finished. " + notes,
             last_metrics=after, finished_at=_now())
    except Exception as exc:  # noqa: BLE001
        if run_id is not None:
            database.finish_run(run_id, "failed", None, str(exc))
        _set(running=False, stage="failed", message=f"Retraining failed: {exc}", finished_at=_now())


if __name__ == "__main__":
    # Command-line retraining: python -m src.retrain
    database.init_db()
    run_retraining()
    print(get_status()["message"])
