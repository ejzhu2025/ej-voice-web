FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download XTTS-v2 model at build time so it's baked into the image
RUN python -c "\
from TTS.tts.configs.xtts_config import XttsConfig; \
from TTS.tts.models.xtts import Xtts; \
import os, shutil; \
from TTS.utils.manage import ModelManager; \
mm = ModelManager(); \
model_path, _, _ = mm.download_model('tts_models/multilingual/multi-dataset/xtts_v2'); \
os.makedirs('/app/xtts_model', exist_ok=True); \
[shutil.copy2(os.path.join(model_path, f), '/app/xtts_model/') for f in os.listdir(model_path)]"

COPY app.py .
COPY ej_en.wav .
COPY templates/ templates/
COPY static/ static/

ENV XTTS_MODEL_PATH=/app/xtts_model

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
