"""
SAM3 Background Replacement — FastAPI Web Server

Usage:
    python server.py                    # default: 0.0.0.0:8080
    python server.py --port 7860        # custom port
    python server.py --host 0.0.0.0     # expose to network (Vast.AI)

Visiting http://<host>:<port>/ redirects automatically to the web UI.
"""

import io
import uuid
import asyncio
import argparse
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse

from cutpaste import replace_background

app = FastAPI(title="SAM3 Background Replacement", docs_url="/docs")

# One GPU inference at a time
_gpu_lock = asyncio.Lock()

TMP_DIR = Path("/tmp/cutpaste_uploads")
TMP_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Web UI (inline HTML — no extra files needed)
# ---------------------------------------------------------------------------

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>SAM3 Background Replace</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #0f1117;
    color: #e2e8f0;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 2rem 1rem;
  }
  h1 {
    font-size: 1.8rem;
    font-weight: 700;
    color: #7c3aed;
    margin-bottom: 0.3rem;
    letter-spacing: -0.02em;
  }
  .subtitle { color: #64748b; font-size: 0.9rem; margin-bottom: 2rem; }
  .card {
    background: #1e2130;
    border: 1px solid #2d3148;
    border-radius: 12px;
    padding: 1.5rem;
    width: 100%;
    max-width: 900px;
    margin-bottom: 1.2rem;
  }
  .card-title {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #7c3aed;
    margin-bottom: 1rem;
    font-weight: 600;
  }
  .uploads {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
  }
  @media (max-width: 600px) { .uploads { grid-template-columns: 1fr; } }
  .drop-zone {
    border: 2px dashed #3d4163;
    border-radius: 10px;
    padding: 1.5rem;
    text-align: center;
    cursor: pointer;
    transition: border-color 0.2s, background 0.2s;
    position: relative;
    min-height: 180px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
  }
  .drop-zone:hover, .drop-zone.dragover {
    border-color: #7c3aed;
    background: #1a1d30;
  }
  .drop-zone input[type=file] {
    position: absolute; inset: 0; opacity: 0; cursor: pointer; width: 100%; height: 100%;
  }
  .drop-zone .icon { font-size: 2rem; margin-bottom: 0.5rem; }
  .drop-zone .label { font-size: 0.85rem; color: #94a3b8; }
  .drop-zone .sublabel { font-size: 0.75rem; color: #475569; margin-top: 0.25rem; }
  .drop-zone img.preview {
    max-width: 100%; max-height: 150px; border-radius: 6px;
    object-fit: contain; margin-top: 0.5rem;
  }
  .drop-zone .filename {
    font-size: 0.75rem; color: #7c3aed; margin-top: 0.4rem; word-break: break-all;
  }
  .settings-toggle {
    background: none; border: none; color: #7c3aed; cursor: pointer;
    font-size: 0.85rem; display: flex; align-items: center; gap: 0.4rem;
    padding: 0; margin-bottom: 0.8rem;
  }
  .settings-toggle:hover { color: #a78bfa; }
  .settings-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 1rem;
  }
  @media (max-width: 600px) { .settings-grid { grid-template-columns: 1fr; } }
  label.field-label { font-size: 0.8rem; color: #94a3b8; display: block; margin-bottom: 0.3rem; }
  input[type=text], input[type=number] {
    width: 100%; background: #0f1117; border: 1px solid #2d3148;
    border-radius: 6px; color: #e2e8f0; padding: 0.5rem 0.75rem; font-size: 0.9rem;
  }
  input:focus { outline: none; border-color: #7c3aed; }
  .btn {
    background: #7c3aed; color: white; border: none; border-radius: 8px;
    padding: 0.75rem 2rem; font-size: 1rem; font-weight: 600;
    cursor: pointer; transition: background 0.2s; width: 100%;
    display: flex; align-items: center; justify-content: center; gap: 0.5rem;
  }
  .btn:hover:not(:disabled) { background: #6d28d9; }
  .btn:disabled { background: #374151; color: #6b7280; cursor: not-allowed; }
  .spinner {
    width: 18px; height: 18px;
    border: 2px solid rgba(255,255,255,0.3);
    border-top-color: white;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    display: none;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .status { font-size: 0.85rem; text-align: center; margin-top: 0.75rem; min-height: 1.2em; }
  .status.error { color: #f87171; }
  .status.info  { color: #94a3b8; }
  .status.ok    { color: #4ade80; }
  .result-area { display: none; text-align: center; }
  .result-area img { max-width: 100%; border-radius: 8px; margin-bottom: 1rem; }
  .btn-download {
    background: #059669; color: white; border: none; border-radius: 8px;
    padding: 0.6rem 1.5rem; font-size: 0.9rem; font-weight: 600;
    cursor: pointer; transition: background 0.2s; display: inline-block; text-decoration: none;
  }
  .btn-download:hover { background: #047857; }
  .divider { border: none; border-top: 1px solid #2d3148; margin: 1rem 0; }
  details > summary { list-style: none; }
  details > summary::-webkit-details-marker { display: none; }
</style>
</head>
<body>
<h1>SAM3 Background Replace</h1>
<p class="subtitle">Segment Anything Model 3 — powered person cutout &amp; background compositing</p>

<!-- Upload Card -->
<div class="card">
  <div class="card-title">Step 1 — Upload Images</div>
  <div class="uploads">
    <div>
      <div style="font-size:0.85rem;color:#94a3b8;margin-bottom:0.5rem;">Portrait / Person Image</div>
      <div class="drop-zone" id="zone-portrait" ondragover="onDrag(event,'zone-portrait')"
           ondragleave="offDrag('zone-portrait')" ondrop="onDrop(event,'portrait')">
        <input type="file" id="inp-portrait" accept="image/*" onchange="onFile(event,'portrait')"/>
        <div class="icon" id="icon-portrait">🧍</div>
        <div class="label" id="label-portrait">Click or drag portrait here</div>
        <div class="sublabel">JPG, PNG, WEBP</div>
        <img id="prev-portrait" class="preview" style="display:none"/>
        <div class="filename" id="fname-portrait"></div>
      </div>
    </div>
    <div>
      <div style="font-size:0.85rem;color:#94a3b8;margin-bottom:0.5rem;">New Background Image</div>
      <div class="drop-zone" id="zone-background" ondragover="onDrag(event,'zone-background')"
           ondragleave="offDrag('zone-background')" ondrop="onDrop(event,'background')">
        <input type="file" id="inp-background" accept="image/*" onchange="onFile(event,'background')"/>
        <div class="icon" id="icon-background">🏞️</div>
        <div class="label" id="label-background">Click or drag background here</div>
        <div class="sublabel">JPG, PNG, WEBP</div>
        <img id="prev-background" class="preview" style="display:none"/>
        <div class="filename" id="fname-background"></div>
      </div>
    </div>
  </div>
</div>

<!-- Settings Card -->
<div class="card">
  <details id="settings-details">
    <summary>
      <button class="settings-toggle" onclick="this.blur()">
        <span id="settings-arrow">▶</span> Advanced Settings
      </button>
    </summary>
    <div class="settings-grid" style="margin-top:0.5rem;">
      <div>
        <label class="field-label" for="inp-prompt">Segmentation Prompt</label>
        <input type="text" id="inp-prompt" value="person" placeholder="person"/>
      </div>
      <div>
        <label class="field-label" for="inp-fill">Person Fill (% of frame height)</label>
        <input type="number" id="inp-fill" value="75" min="10" max="100" step="5"/>
      </div>
      <div>
        <label class="field-label" for="inp-feather">Edge Feather (σ)</label>
        <input type="number" id="inp-feather" value="3.0" min="0.0" max="20.0" step="0.5"/>
      </div>
      <div>
        <label class="field-label" for="inp-conf">Confidence Threshold</label>
        <input type="number" id="inp-conf" value="0.5" min="0.0" max="1.0" step="0.05"/>
      </div>
    </div>
    <div style="margin-top:1rem;display:flex;align-items:center;gap:0.6rem;">
      <input type="checkbox" id="inp-harmonize" checked
             style="width:16px;height:16px;accent-color:#7c3aed;cursor:pointer;"/>
      <label for="inp-harmonize" style="font-size:0.85rem;color:#e2e8f0;cursor:pointer;">
        PCTNet Harmonization
        <span style="color:#64748b;font-size:0.78rem;margin-left:0.3rem;">
          — adjusts person lighting &amp; colors to match background
        </span>
      </label>
    </div>
  </details>
</div>

<!-- Process Card -->
<div class="card">
  <div class="card-title">Step 2 — Process</div>
  <button class="btn" id="btn-process" onclick="processImages()" disabled>
    <div class="spinner" id="spinner"></div>
    <span id="btn-text">Replace Background</span>
  </button>
  <div class="status info" id="status">Upload both images to begin</div>
</div>

<!-- Result Card -->
<div class="card result-area" id="result-card">
  <div class="card-title">Result</div>
  <img id="result-img" src="" alt="Result"/>
  <a id="download-link" class="btn-download" download="result.png">Download PNG</a>
</div>

<script>
  const files = { portrait: null, background: null };

  function onDrag(e, zoneId) {
    e.preventDefault();
    document.getElementById(zoneId).classList.add('dragover');
  }
  function offDrag(zoneId) {
    document.getElementById(zoneId).classList.remove('dragover');
  }
  function onDrop(e, key) {
    e.preventDefault();
    offDrag('zone-' + key);
    const file = e.dataTransfer.files[0];
    if (file) setFile(key, file);
  }
  function onFile(e, key) {
    const file = e.target.files[0];
    if (file) setFile(key, file);
  }
  function setFile(key, file) {
    files[key] = file;
    document.getElementById('fname-' + key).textContent = file.name;
    document.getElementById('icon-' + key).style.display = 'none';
    document.getElementById('label-' + key).style.display = 'none';
    const prev = document.getElementById('prev-' + key);
    prev.src = URL.createObjectURL(file);
    prev.style.display = 'block';
    checkReady();
  }
  function checkReady() {
    const ready = files.portrait && files.background;
    document.getElementById('btn-process').disabled = !ready;
    if (ready) setStatus('Ready — click "Replace Background"', 'info');
  }
  function setStatus(msg, type) {
    const el = document.getElementById('status');
    el.textContent = msg;
    el.className = 'status ' + type;
  }
  function setLoading(loading) {
    document.getElementById('spinner').style.display = loading ? 'block' : 'none';
    document.getElementById('btn-text').textContent = loading ? 'Processing on GPU…' : 'Replace Background';
    document.getElementById('btn-process').disabled = loading;
  }

  async function processImages() {
    if (!files.portrait || !files.background) return;
    setLoading(true);
    const useHarm = document.getElementById('inp-harmonize').checked;
    setStatus(useHarm ? 'Running SAM3 + PCTNet harmonization…' : 'Running SAM3 segmentation…', 'info');
    document.getElementById('result-card').style.display = 'none';

    const form = new FormData();
    form.append('portrait', files.portrait);
    form.append('background', files.background);
    form.append('prompt', document.getElementById('inp-prompt').value || 'person');
    form.append('confidence', document.getElementById('inp-conf').value);
    form.append('feather', document.getElementById('inp-feather').value);
    form.append('person_fill', document.getElementById('inp-fill').value / 100);
    form.append('harmonize', document.getElementById('inp-harmonize').checked ? '1' : '0');

    try {
      const res = await fetch('/api/process', { method: 'POST', body: form });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || res.statusText);
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      document.getElementById('result-img').src = url;
      document.getElementById('download-link').href = url;
      document.getElementById('result-card').style.display = 'block';
      setStatus('Done! Scroll down to see result.', 'ok');
      document.getElementById('result-card').scrollIntoView({ behavior: 'smooth' });
    } catch (err) {
      setStatus('Error: ' + err.message, 'error');
    } finally {
      setLoading(false);
      checkReady();
    }
  }

  document.getElementById('settings-details').addEventListener('toggle', function() {
    document.getElementById('settings-arrow').textContent = this.open ? '▼' : '▶';
  });
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    """Redirect bare URL to the web UI."""
    return RedirectResponse(url="/ui", status_code=302)


@app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
async def ui():
    return _HTML


@app.post("/api/process")
async def process(
    portrait: UploadFile = File(..., description="Portrait/person image"),
    background: UploadFile = File(..., description="New background image"),
    prompt: str = Form("person"),
    confidence: float = Form(0.5),
    feather: float = Form(3.0),
    person_fill: float = Form(0.75),
    harmonize: int = Form(1),
):
    """
    Segment the person from `portrait` using SAM3 and composite onto `background`.
    Returns the result as a PNG download.
    """
    uid = uuid.uuid4().hex[:10]
    portrait_suffix = Path(portrait.filename or "img.jpg").suffix or ".jpg"
    background_suffix = Path(background.filename or "img.jpg").suffix or ".jpg"

    portrait_path = TMP_DIR / f"{uid}_portrait{portrait_suffix}"
    background_path = TMP_DIR / f"{uid}_background{background_suffix}"
    output_path = TMP_DIR / f"{uid}_output.png"

    portrait_path.write_bytes(await portrait.read())
    background_path.write_bytes(await background.read())

    async with _gpu_lock:
        try:
            await asyncio.to_thread(
                replace_background,
                str(portrait_path),
                str(background_path),
                str(output_path),
                prompt=prompt,
                confidence_threshold=confidence,
                feather_sigma=feather,
                person_fill=person_fill,
                harmonize=bool(harmonize),
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Inference error: {exc}")
        finally:
            portrait_path.unlink(missing_ok=True)
            background_path.unlink(missing_ok=True)

    output_bytes = output_path.read_bytes()
    output_path.unlink(missing_ok=True)

    return StreamingResponse(
        io.BytesIO(output_bytes),
        media_type="image/png",
        headers={"Content-Disposition": f'attachment; filename="result_{uid}.png"'},
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM3 Background Replacement Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  SAM3 Background Replacement Server")
    print("=" * 55)
    print(f"  Web UI  : http://{args.host}:{args.port}/")
    print(f"  API doc : http://{args.host}:{args.port}/docs")
    print("=" * 55)
    print("  Visiting / auto-redirects to the web UI.")
    print("  On Vast.AI: use the port-forwarded URL.")
    print("=" * 55 + "\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
