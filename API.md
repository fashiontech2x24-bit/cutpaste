# API

`POST /api/process` — replace background, returns PNG

## Parameters (multipart/form-data)

| Field | Type | Default | Description |
|---|---|---|---|
| `portrait` | file | required | Person image |
| `background` | file | required | New background |
| `prompt` | string | `person` | Segmentation prompt |
| `confidence` | float | `0.5` | Detection threshold (0–1) |
| `feather` | float | `1.5` | Edge feather σ |
| `erode` | int | `2` | Edge erosion (px) |
| `person_fill` | float | `0.92` | Body height as fraction of frame |
| `foot_anchor` | float | `0.92` | Feet position as fraction from top |

## Example

```python
import requests

with open("portrait.jpg", "rb") as p, open("bg.jpg", "rb") as b:
    r = requests.post("http://localhost:8000/api/process", files={
        "portrait": p, "background": b
    })
    open("result.png", "wb").write(r.content)
```

```bash
curl -X POST http://localhost:8000/api/process \
  -F portrait=@portrait.jpg \
  -F background=@bg.jpg \
  -o result.png
```

## Output

768×1024 PNG. Errors return JSON `{"detail": "..."}` with status 422 or 500.
