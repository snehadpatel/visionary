# Visionary — AI-Powered Room Redesign

Smart AI-powered room redesign web application that uses a custom Vision-Language Model (VLM) pipeline to understand rooms, generate photorealistic redesigns, allocate budgets intelligently, and match real products from Indian e-commerce.

## Architecture

```
Room Image → CLIP ViT-L/14 → Projection MLP → TinyLlama-1.1B → Structured Analysis
                                                                      ↓
                                     ← SD v1.5 + ControlNet ← SD Prompt Builder
                                                                      ↓
                                                               Budget Engine → Product Scraper
```

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18 + Tailwind CSS v4 + Framer Motion |
| Backend | FastAPI (Python 3.11) |
| VLM — Vision Encoder | CLIP ViT-L/14 (frozen) |
| VLM — Language Decoder | TinyLlama-1.1B-Chat |
| VLM — Projection Layer | Custom 2-layer MLP (trained from scratch) |
| Object Detection | YOLOv8x (ultralytics) |
| Segmentation | SAM vit_h (Segment Anything) |
| Depth Estimation | MiDaS DPT_Large |
| Style Classification | CLIP zero-shot |
| Image Generation | Stable Diffusion v1.5 + ControlNet Canny |
| Budget Engine | Priority-weighted allocation (Pure Python) |
| Product Matching | Web scraping — Amazon.in, Pepperfry |
| Runtime | Apple MPS (M4 GPU) |

## Setup & Run

```bash
# 1. Download SAM weights (2.5GB)
mkdir -p models
wget -P models/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# 2. Install backend dependencies (in visionary_env)
source visionary_env/bin/activate
pip install -r backend/requirements.txt

# 3. Install frontend dependencies
cd frontend && npm install && cd ..

# 4. (Optional) Train the projection MLP
cd training && python train_projection.py && cd ..

# 5. Start backend
cd backend && python main.py

# 6. Start frontend (new terminal)
cd frontend && npm run dev
```

**Models auto-download on first run:**
- YOLOv8x: ~130MB
- MiDaS DPT_Large: ~400MB
- CLIP ViT-L/14: ~890MB
- TinyLlama-1.1B: ~2.2GB
- SD v1.5: ~4.0GB
- ControlNet Canny: ~1.4GB

**Total disk: ~9GB | Minimum 16GB unified memory recommended**

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/redesign` | Start room redesign (image + budget + style) |
| GET | `/api/status/{job_id}` | Poll pipeline progress |
| POST | `/api/chat/{job_id}` | Conversational design refinement |
| GET | `/api/products/{job_id}` | Get matched products |
| GET | `/health` | Health check |

## Project Structure

```
visionary/
├── backend/
│   ├── main.py              # FastAPI entry point
│   ├── routers/             # API route handlers
│   ├── vlm/                 # Custom VLM (CLIP + MLP + TinyLlama)
│   ├── pipeline/            # CV pipeline modules
│   ├── products/            # Scraping & matching
│   └── utils/               # Storage & image utilities
├── frontend/
│   └── src/
│       ├── components/      # React components
│       ├── hooks/           # Custom React hooks
│       └── api/             # API client
├── training/                # VLM fine-tuning scripts
├── models/                  # Model weights
└── outputs/                 # Generated images
```

## No External APIs

Zero paid API dependencies. Every model runs locally on Apple MPS.
