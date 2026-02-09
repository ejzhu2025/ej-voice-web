import io
import os
import torch
import scipy.io.wavfile
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

SPEAKER_WAV = os.path.join(os.path.dirname(__file__), "ej_en.wav")
MODEL_PATH = os.environ.get(
    "XTTS_MODEL_PATH",
    os.path.expanduser("~/Library/Application Support/tts/tts_models--multilingual--multi-dataset--xtts_v2"),
)

model = None
gpt_cond_latent = None
speaker_embedding = None


def load_model():
    global model, gpt_cond_latent, speaker_embedding
    print("Loading XTTS-v2 model...")
    config = XttsConfig()
    config.load_json(os.path.join(MODEL_PATH, "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=MODEL_PATH, eval=True)
    model.cpu()
    print("Computing speaker embedding...")
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=[SPEAKER_WAV], gpt_cond_len=12
    )
    print("Model ready.")


@app.on_event("startup")
async def startup():
    load_model()


class SynthesizeRequest(BaseModel):
    text: str


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/synthesize")
async def synthesize(req: SynthesizeRequest):
    if not req.text or not req.text.strip():
        return JSONResponse(status_code=400, content={"error": "Text is required"})

    text = req.text.strip()[:500]  # limit length

    out = model.inference(
        text,
        "en",
        gpt_cond_latent,
        speaker_embedding,
        temperature=0.3,
    )

    wav = torch.tensor(out["wav"]).unsqueeze(0)
    sample_rate = 24000

    buf = io.BytesIO()
    scipy.io.wavfile.write(buf, sample_rate, wav.squeeze().numpy())
    buf.seek(0)

    return StreamingResponse(buf, media_type="audio/wav")
