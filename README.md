# AI-Public-Speaking_-SWARA

FastAPI-based audio analysis service for the **SWARA** (public-speaking AI) platform.  
Deployed on Hugging Face Spaces: <https://huggingface.co/spaces/Cyberlace/api-swara-audio-analysis>

---

## Features

| Analysis | Description |
|---|---|
| **Tempo** | Speech rate (WPM) and long-pause detection |
| **Articulation** | Filler-word count and clarity score |
| **Profanity** | Inappropriate language detection |
| **Keywords** | Topic keyword coverage (optional) |
| **Structure** | Content structure / reference-text similarity (optional) |

---

## API Reference

### POST `/api/v1/analyze`

Submit an audio file for asynchronous analysis.

**Form fields**

| Field | Type | Required | Description |
|---|---|---|---|
| `audio` | file (WAV) | ✅ | The audio recording to analyse |
| `custom_topic` | string | ❌ | Topic label used as fallback keywords |
| `reference_text` | string | ❌ | Reference/script text for structure scoring |
| `custom_keywords` | JSON string (array) | ❌ | Explicit keywords, e.g. `'["kata1","kata2"]'` |

**Response**

```json
{ "task_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6" }
```

---

### GET `/api/v1/status/{task_id}`

Poll the result of a previously submitted analysis.

**Response**

```json
{
  "task_id": "3fa85f64-…",
  "status": "completed",
  "result": {
    "tempo": {
      "score": 2,
      "has_long_pause": false,
      "wpm": 130.5,
      "duration_sec": 45.2
    },
    "articulation": {
      "score": 1,
      "filler_count": 3,
      "fillers_found": ["um", "uh", "anu"]
    },
    "profanity": {
      "has_profanity": false,
      "profanity_words": []
    },
    "keywords": {
      "score": 2,
      "keywords_found": ["komunikasi", "kepemimpinan"],
      "coverage": 1.0
    },
    "structure": {
      "score": 1,
      "similarity": 0.42
    },
    "transcript": "Selamat pagi, hari ini saya akan membahas…"
  },
  "error": null
}
```

`status` can be `"pending"`, `"completed"`, or `"failed"`.  
`result` is `null` while the task is still pending.

**Scoring scale** (per dimension): `0` = poor · `1` = acceptable · `2` = good

---

## Local Development

```bash
# Install dependencies
pip install -r app/requirements.txt

# Run the server
uvicorn app.main:app --reload --port 7860
```

Interactive docs available at <http://localhost:7860/docs>.

---

## Docker

```bash
docker build -t swara-audio-api .
docker run -p 7860:7860 swara-audio-api
```

---

## Project Structure

```
.
├── app/
│   ├── main.py            # FastAPI application & task management
│   ├── audio_analysis.py  # Audio & text analysis logic
│   └── requirements.txt   # Python dependencies
├── Dockerfile
└── README.md
```