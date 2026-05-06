---
title: Swara API - Audio Analysis
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# Swara API - Audio Analysis Service 🎙️

AI-powered audio analysis service untuk penilaian public speaking.

## Features

- 🎤 Speech-to-Text dengan Whisper
- ⏱️ Tempo & Jeda Analysis
- 🗣️ Articulation Assessment
- 📊 Structure Detection
- 🔍 Keyword Relevance Analysis

## API Documentation

Once deployed, visit:

- `/docs` - Interactive Swagger UI
- `/redoc` - ReDoc documentation
- `/api/v1/health` - Health check

## Usage

```bash
# Submit audio for analysis
curl -X POST "https://YOUR_SPACE.hf.space/api/v1/analyze" \
  -F "audio=@your_audio.wav" \
  -F "analyze_tempo=true" \
  -F "analyze_structure=true"
```

For detailed documentation, see the full README in the repository.
