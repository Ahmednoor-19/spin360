# Spin360

Two product photos (front + back) → a seamless 3–5 s **360° spin video (MP4)**, fully automated.

Pipeline:

```
upload → isolate → reconstruct (front+back → GLB) → normalize → render → encode → quality-gate → MP4
```

## Demo

Production path (`fal_trellis_multi` reconstruction + Blender Cycles render):

<video src="https://raw.githubusercontent.com/Ahmednoor-19/spin360/main/assets/demo.mp4" controls width="360"></video>

## Why it runs anywhere (and where the real fidelity comes from)

The one generative step (3D reconstruction) sits behind a **swappable provider
interface**, and the renderer has two backends. So the same code runs in three modes:

| | Reconstruction | Render | Needs |
|---|---|---|---|
| **Demo / CI** (default) | `mock` (silhouette-extruded slab, real geometry) | `cpu` (numpy rasterizer) | nothing but Python + ffmpeg |
| **Production** | `fal_trellis_multi` (fal.ai TRELLIS multi-view) | `blender` (headless EEVEE) | `FAL_KEY` + Blender + GPU |
| **Self-hosted** | drop-in provider (same interface) | `blender` | your GPU box |

The mock produces a genuinely spinning, textured, quality-gated MP4 with **no GPU
and no network** — so the whole architecture is demonstrable and testable today.
Swap two env vars and the identical pipeline runs the high-fidelity path.

## Quickstart (demo mode, no infra)

```bash
pip install -r requirements.txt          # ffmpeg must also be on PATH
python scripts/make_samples.py samples   # synthetic front/back product photos
python scripts/run_local.py samples/front.png samples/back.png
```

## Production mode (API + worker + queue)

```bash
cp .env.example .env      # set FAL_KEY, SPIN360_RECON_PROVIDER=fal_trellis_multi, SPIN360_RENDER=blender
docker compose up --build
# POST two images:
curl -F front_image=@samples/front.png -F back_image=@samples/back.png \
     -F duration_s=4 http://localhost:8000/jobs           # -> {job_id, status:queued, ...}
curl http://localhost:8000/jobs/<job_id>                  # poll the JobRecord
curl -o spin.mp4 http://localhost:8000/jobs/<job_id>/video
```

## UI

`GET /` serves a single-page studio UI (no build step, no external assets): drop a
front and a back photo, hit **Spin it**, and a circular turntable dial fills 0→360°
as the job moves through the six stages in plain language ("Lifting it off the
background", "Filming the turn"), then the finished loop plays inside the dial with
a quality score and a download button. It talks to the same `POST /jobs` → poll →
`/video` contract below. In demo mode jobs run in a background thread so the dial
shows **live** stage progress.

## Run on a free GPU (Google Colab)

`Spin360_Colab.ipynb` runs the whole thing on a free T4 — no fal key, no local GPU:

1. Open the notebook in Colab, set Runtime → **T4 GPU**, upload `spin360.zip`.
2. It installs deps + headless **Blender** (Cycles renders on the GPU without a display).
3. It launches the API and prints a **public Colab link** to the pitch UI (mock
   reconstruction + GPU render — instant), plus a one-shot cell that renders your own
   two photos inline.
4. An optional cell installs **self-hosted TRELLIS** on the T4 for *real* front+back 3D
   (heavier setup, no fal key needed) — then re-run the demo for full fidelity.

## Reconstruction backends

Selected by `SPIN360_RECON_PROVIDER`, all behind one interface (`ReconstructProvider`):

| value | what it is | needs |
|---|---|---|
| `mock` *(default)* | silhouette-extruded textured slab, real geometry | nothing |
| `local_trellis` | self-hosted microsoft/TRELLIS multi-image (front+back) | a CUDA GPU (e.g. Colab T4) |
| `fal_trellis_multi` | hosted TRELLIS via fal.ai | `FAL_KEY` |

Multi-image fusion (front+back) uses the **original** TRELLIS (`run_multi_image`);
TRELLIS.2 is single-image only, so it can't combine two views.

## Tests

```bash
python -m pytest -q      # schema contract, idempotency, budget guard, full e2e (mock+cpu)
```

## Layout

```
spin360/
  config.py          pinned versions + budgets
  schemas.py         JobStatus + JobRecord
  db.py storage.py   truth store + object storage
  queue.py worker.py async job queue
  reliability.py     idempotency / retry / breaker / budget
  observability.py   tracing + per-stage timings
  orchestrator.py    the sequential controller
  api.py             FastAPI: submit / poll / fetch video
  web/index.html     single-page pitch UI (turntable dial)
  pipeline/          the six stages
  blender/turntable.py  headless production renderer — Cycles GPU or EEVEE
scripts/             make_samples.py, run_local.py
tests/               unit + end-to-end
Spin360_Colab.ipynb 
```
