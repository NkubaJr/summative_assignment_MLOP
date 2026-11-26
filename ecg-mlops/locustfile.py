import glob
import os
import random

from locust import HttpUser, between, task

# Use real test images so every request exercises the full preprocessing + model path
IMAGES = sorted(glob.glob(os.path.join("data", "test", "*", "*.jpg")))
random.seed(42)
SAMPLE = random.sample(IMAGES, min(40, len(IMAGES)))
CACHE = {p: open(p, "rb").read() for p in SAMPLE}


class EcgUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(10)
    def predict(self):
        path = random.choice(SAMPLE)
        self.client.post(
            "/predict",
            files={"file": (os.path.basename(path), CACHE[path], "image/jpeg")},
            name="/predict",
        )

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
