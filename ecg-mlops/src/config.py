import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(BASE_DIR, "data"))
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
FEATURE_CACHE_PATH = os.path.join(DATA_DIR, "features_cache.npz")

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(BASE_DIR, "models"))
ARCHIVE_DIR = os.path.join(MODEL_DIR, "archive")
MODEL_PATH = os.path.join(MODEL_DIR, "ecg_hog_mlp.pkl")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")
METRICS_PATH = os.path.join(MODEL_DIR, "test_metrics.json")

DB_PATH = os.environ.get("DB_PATH", os.path.join(DATA_DIR, "app.db"))

# Size of the cleaned trace image (height, width) that HOG features are computed on
TRACE_SIZE = (256, 384)
SEED = 42

# Retraining is triggered automatically when this many new labelled images are waiting
RETRAIN_THRESHOLD = int(os.environ.get("RETRAIN_THRESHOLD", "20"))
