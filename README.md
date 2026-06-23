# Sugumi — 1.58-bit Large Language Model

**Sugumi** is a ~1.58 billion parameter decoder-only language model that uses **BitNet b1.58 ternary quantization** (`{-1, 0, +1}` weights) to deliver **memory efficiency**, **inference speed**, and **competitive accuracy** — making powerful AI accessible on laptops, phones, and edge devices without cloud GPUs.

> Built as an 8-semester research & engineering project. This repository contains **full source code** for architecture, quantization, training, inference, and chatbot deployment — designed for **GitHub showcase and technical interviews** without requiring high-RAM local execution.

---

## Why 1.58-bit?

| Metric | FP16 (standard) | Sugumi 1.58-bit | Benefit |
|--------|-----------------|-----------------|---------|
| Bits per weight | 16 | ~1.585 (`log₂3`) | ~10× smaller weights |
| ~1.58B model RAM | ~3.2 GB | ~**0.31 GB** | Runs on consumer hardware |
| CPU inference | Slow (FP ops) | Fast (integer-like ops) | Edge & offline chatbots |
| Accuracy | Baseline | Near-baseline with STE training | Practical for real apps |

**1.58-bit** is not arbitrary — ternary values need exactly `log₂(3) ≈ 1.585` bits per weight, as introduced in Microsoft's [BitNet b1.58 paper](https://arxiv.org/abs/2402.17764).

---

## How End Users Are Helped

| User | Problem today | How Sugumi helps |
|------|---------------|------------------|
| **Students / developers** | Can't run 7B+ models locally | Chat on a laptop with <512 MB model RAM |
| **Rural / low-bandwidth users** | Cloud AI needs constant internet | Offline chatbot on phone or Raspberry Pi |
| **Small businesses** | API costs scale with usage | Self-host one lightweight model for all support tickets |
| **Healthcare / legal (privacy)** | Data leaves device to OpenAI | On-device inference — data never uploaded |
| **IoT / embedded** | No GPU on device | 1.58-bit weights fit in flash memory |

---

## Project Structure

```
1.58_bit_llm/
├── sugumi/
│   ├── model/           # Transformer architecture (GQA, RoPE, SwiGLU)
│   ├── quantization/    # BitNet b1.58 ternary {-1,0,+1} + STE
│   ├── inference/       # Inference engine (mock + live modes)
│   ├── chatbot/         # FastAPI REST API + CLI chatbot
│   └── training/        # Cloud GPU training loop
├── configs/
│   └── sugumi_1.58b.yaml
├── scripts/
│   ├── showcase_stats.py   # ✅ Safe on any laptop — prints memory stats
│   ├── run_chatbot.py      # ✅ Mock API server — no weights needed
│   └── train.py            # ⚠️  Cloud GPU only
├── docs/
│   ├── ARCHITECTURE.md
│   ├── CHATBOT_GUIDE.md
│   └── END_USER_BENEFITS.md
├── examples/
│   └── chat_demo.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Quick Start (Showcase Mode — No GPU Required)

These commands are **safe on a low-RAM laptop** — they use mock mode and only demonstrate architecture:

```bash
# Clone your repo
git clone https://github.com/YOUR_USERNAME/sugumi-llm.git
cd sugumi-llm

# Optional: create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

pip install -r requirements.txt

# 1. Print memory & compression stats (no model load)
python scripts/showcase_stats.py

# 2. Interactive CLI chatbot (mock mode — no weights)
python -m sugumi.chatbot.cli --mock

# 3. REST API chatbot (mock mode)
python scripts/run_chatbot.py --mock
# Open http://127.0.0.1:8000/docs for Swagger UI
```

### Example API call (mock mode)

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is 1.58-bit quantization?"}]}'
```

---

## Full Training & Inference (Cloud GPU)

Do **not** run training on a low-RAM laptop. Use Google Colab, AWS, Lambda Labs, or similar:

```bash
# On cloud GPU machine
python scripts/train.py --output_dir ./checkpoints/sugumi-1.58b --epochs 3 --device cuda

# Inference with trained checkpoint
python -m sugumi.chatbot.cli --mock=False --checkpoint ./checkpoints/sugumi-1.58b/sugumi_158b.pt
```

---

## Building Chatbots with Sugumi

Three integration paths:

1. **CLI chatbot** — `python -m sugumi.chatbot.cli` for terminal demos
2. **REST API** — `FastAPI` server compatible with frontend apps (React, Flutter, etc.)
3. **Python SDK** — embed `SugumiEngine` directly in your app

See [docs/CHATBOT_GUIDE.md](docs/CHATBOT_GUIDE.md) for full examples including web UI integration.

```python
from sugumi.inference.engine import SugumiEngine

engine = SugumiEngine(mock=True)  # set mock=False + checkpoint on GPU
reply = engine.chat([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain 1.58-bit LLMs briefly."},
])
print(reply)
```

---

## Architecture Highlights (Interview Talking Points)

- **BitLinear158** — ternary weight quantization with Straight-Through Estimator (STE)
- **GQA** (Grouped Query Attention) — fewer KV heads → faster inference, less KV-cache RAM
- **RoPE** positional embeddings — better long-context generalization
- **SwiGLU FFN** — modern feed-forward used in LLaMA/Mistral-class models
- **8-bit activation quantization** — additional memory savings during training
- **KV-cache** in `generate()` — O(1) per-token cost after prefill

Full details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

---

## Requirements

| Component | Minimum (showcase) | Full training |
|-----------|-------------------|---------------|
| Python | 3.10+ | 3.10+ |
| RAM | 4 GB | 32 GB+ |
| GPU | Not required (mock) | NVIDIA A100/L4 24GB+ |
| Disk | 500 MB (code only) | 50 GB+ (data + checkpoints) |
| OS | Windows / Linux / Mac | Linux preferred |

Install: `pip install -r requirements.txt`

---

## 8-Semester Development Timeline

| Semester | Focus |
|----------|-------|
| 1–2 | LLM fundamentals, transformer math, PyTorch |
| 3 | Quantization survey (INT8, INT4, binary, ternary) |
| 4 | BitNet b1.58 implementation & unit tests |
| 5 | Model scaling to ~1.58B params, config tuning |
| 6 | Training pipeline, distillation from teacher model |
| 7 | Inference optimization, chatbot API |
| 8 | Edge deployment, documentation, benchmark report |

---

## License

MIT License — see [LICENSE](LICENSE).

---

## Citation

If you reference this project:

```bibtex
@software{sugumi2024,
  title  = {Sugumi: A 1.58-bit Large Language Model},
  author = {Sugumi Project},
  year   = {2024},
  url    = {https://github.com/YOUR_USERNAME/sugumi-llm}
}
```

Based on: *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits* (Ma et al., 2024).
