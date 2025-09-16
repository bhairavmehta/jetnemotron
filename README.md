# Jet-Nemotron (Minimal Reconstruction)

This repository provides a **minimal PyTorch scaffold** inspired by the ideas in the *Jet-Nemotron* paper from NVLabs. It is **not the official implementation**, but a reconstruction based on the paper details to help researchers and practitioners experiment with the architecture concepts.

---

## 📖 Overview

Jet-Nemotron introduces two core ideas:

1. **JetBlocks** – Transformer blocks with **switchable sublayers**, allowing attention types (MHA, linear-attn), FFN types (SwiGLU, GatedMLP), and optional local convolutional mixing.

2. **PostNAS** – A *late-stage neural architecture search* approach that selects the best block variants **after initial pretraining** using a proxy validation set.

This repo provides:
- A clean **PyTorch Jet-Nemotron skeleton**
- **Attention & FFN variants** (MHA, linear attention, SwiGLU, GatedMLP)
- **Rotary embeddings**
- **Optional local depthwise conv mixing**
- **PostNAS selection loop** for late block replacement
- Toy training + proxy perplexity evaluation script

---

## ⚠️ Disclaimer
- This is **not** the official NVLabs release.
- Official code & pretrained models are pending release ([NVLabs/Jet-Nemotron](https://github.com/NVlabs/Jet-Nemotron)).
- This repo is for **educational and research prototyping** only.

---

## 🚀 Quickstart

### 1. Clone & Install
```bash
pip install torch
```

### 2. Run Demo
```bash
python jet_nemotron.py
```

This runs:
- Warmup training on toy data
- PostNAS candidate search
- Post-tuning
- Greedy sample generation

Expected output:
```
warmup step 5: loss=10.23
[PostNAS] Layer 0: chose {...} -> proxy PPL=123.45
...
Chosen candidates per layer: [...]
post-tune done; proxy ppl: 85.67
sample ids: [123, 456, ...]
```

---

## 🔧 Components

- **`JetBlock`** – switchable block containing:
  - Multi-Head Attention (MHA) or Linear Attention
  - SwiGLU or Gated MLP feedforward
  - Optional depthwise convolution mixing

- **`JetNemotron`** – minimal transformer model with JetBlocks, tied embeddings, and generation loop.

- **`postnas_select`** – greedy per-layer PostNAS implementation using proxy perplexity.

---

## 📂 Structure

```
jet_nemotron.py   # main scaffold (blocks, model, postNAS, training)
README.md         # this file
```

---

## 🛠️ Extend This
- Swap in **FlashAttention / Performer** for efficiency
- Add **grouped-query or multi-query attention**
- Replace toy proxy PPL with **task-specific validation loss**
- Integrate with **HuggingFace Datasets & Tokenizers**
- Add YAML/CLI config system

---

## 📚 References
- [Jet-Nemotron Paper (arXiv)](https://arxiv.org/abs/2508.15884)
- [NVLabs GitHub (placeholder)](https://github.com/NVlabs/Jet-Nemotron)

---

## 🙌 Acknowledgements
Inspired by the NVLabs *Jet-Nemotron* paper. This reconstruction aims to make the core ideas more accessible until the official code release.
