# FlashHead Server

FastAPI service that wraps the SoulX-FlashHead talking-head model and exposes
it as a REST API with async job-based video generation.

## Directory layout

```
server/
├── src/flashhead_server/
│   ├── contracts.py          # GenerationRequest, JobRecord, enums
│   ├── config.py             # Settings (env-var driven)
│   ├── app.py                # FastAPI app factory
│   ├── main.py               # Uvicorn entry-point
│   ├── providers/
│   │   ├── base.py           # TalkingHeadProvider Protocol
│   │   └── flashhead_provider.py  # Real inference wrapper
│   └── routes/
│       ├── system.py         # GET /health, GET /info
│       └── generation.py     # POST /api/generate, GET /api/jobs/{id}, GET /api/jobs/{id}/video
├── tests/
│   └── test_generation.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Prerequisites

Install model runtime deps from the repo root (`SoulX-FlashHead/`):

```bash
pip install -r requirements.txt          # flash_head + torch + diffusers etc.
```

Then install the server package from the `server/` directory:

```bash
cd server
pip install -e ".[dev]"
```

**Important:** the `flash_head` package must be importable. Since it lives in
`SoulX-FlashHead/flash_head/` without a package install, either run the server
from the `SoulX-FlashHead/` root (recommended) or add it to `PYTHONPATH`:

```bash
# Option A — run from repo root (recommended)
cd SoulX-FlashHead
flashhead-server

# Option B — explicit PYTHONPATH
PYTHONPATH=/path/to/SoulX-FlashHead flashhead-server

# Option C — uvicorn directly
cd SoulX-FlashHead
PYTHONPATH=. uvicorn flashhead_server.main:app --host 127.0.0.1 --port 8002
```

## Required environment variables

| Variable | Default | Description |
|---|---|---|
| `FLASHHEAD_CKPT_DIR` | *(required)* | Path to SoulX-FlashHead-1_3B checkpoint directory |
| `FLASHHEAD_WAV2VEC_DIR` | *(required)* | Path to wav2vec2-base-960h checkpoint directory |
| `FLASHHEAD_MODEL_MODE` | `lite` | Model mode: `lite` or `pro` |
| `FLASHHEAD_HOST` | `127.0.0.1` | Bind address |
| `FLASHHEAD_PORT` | `8002` | Bind port |
| `FLASHHEAD_USE_FACE_CROP` | `true` | Auto-detect and crop face from portrait |
| `FLASHHEAD_OUTPUT_DIR` | `server/outputs` | Directory for job artifacts |
| `FLASHHEAD_ENV` | `development` | `development` or `production` |
| `FLASHHEAD_ROOT` | auto-detected | Absolute path to `SoulX-FlashHead/` root |

### Typical startup

```bash
cd SoulX-FlashHead
export FLASHHEAD_CKPT_DIR=/data/models/SoulX-FlashHead-1_3B
export FLASHHEAD_WAV2VEC_DIR=/data/models/wav2vec2-base-960h
export FLASHHEAD_MODEL_MODE=lite
flashhead-server
```

## API

### `GET /health`

```json
{
  "status": "ok",
  "provider": "flashhead",
  "provider_ready": true,
  "model_mode": "lite"
}
```

### `GET /info`

Returns service metadata and configuration.

### `POST /api/generate`

Multipart form upload. Accepts a portrait image and audio file; queues a
background generation job.

**Form fields:**

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `face_image` | file | yes | — | Portrait image (JPEG/PNG) |
| `audio` | file | yes | — | Audio file (WAV, 16 kHz mono) |
| `mode` | string | no | `lite` | `lite` or `pro` |
| `use_face_crop` | bool | no | `true` | Auto-crop face to 512×512 |
| `audio_encode_mode` | string | no | `stream` | `stream` or `once` |
| `base_seed` | int | no | `42` | RNG seed |

**202 response:**

```json
{ "job_id": "uuid", "status": "pending" }
```

### `GET /api/jobs/{job_id}`

```json
{
  "job_id": "...",
  "status": "completed",
  "mode": "lite",
  "created_at": 1714000000.0,
  "updated_at": 1714000120.0,
  "video_ready": true,
  "duration_seconds": 118.4,
  "error_message": null
}
```

**Statuses:** `pending` → `processing` → `completed` | `failed`

### `GET /api/jobs/{job_id}/video`

Downloads the generated MP4. Returns 404 if job is not yet completed.

## Running tests

Tests use a stub provider — no GPU or model weights required.

```bash
cd server
pytest
```

## Model modes

| Mode | Speed | Quality | Notes |
|---|---|---|---|
| `lite` | Fast | Good | Recommended for L4 / lower-end GPUs |
| `pro` | Slow | High | Requires high VRAM, more inference steps |

## Notes

- Model weights are loaded **once at startup** and shared across all requests.
- Only one generation job runs at a time per process (background task queue).
  For concurrency, run multiple server instances.
- Input audio must be 16 kHz mono WAV for best results; librosa will resample
  other formats automatically.
- `use_face_crop=true` (default) detects the face in the portrait, crops it to
  512×512, and composites generated frames back into the original resolution.
