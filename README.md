<div align="center">
  <img src="assets/tiny-speculators.png" alt="tiny-speculators" width="420">

  <h1>tiny-speculators</h1>

  <p><strong>A minimal repository for training speculative decoding models from scratch.</strong></p>

  <p>
    <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white" alt="Python 3.11+">
    <img src="https://img.shields.io/badge/PyTorch-2.9%E2%80%932.12-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 2.9–2.12">
    <img src="https://img.shields.io/badge/vLLM-0.25.1-4051B5?logo=vllm&logoColor=white" alt="vLLM 0.25.1">
    <img src="https://img.shields.io/badge/NVIDIA%20CUDA-13.0-76B900?logo=nvidia&logoColor=white" alt="NVIDIA CUDA 13.0">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  </p>
</div>

---

**tiny-speculators** is a minimal training repository currently supporting EAGLE-3 (Qwen3-8B), which includes the vLLM integration required to train and export a usable draft model.

## EAGLE-3 Draft Model Architecture

First of all, what is a EAGLE Draft Model?

![EAGLE-3 draft model architecture](image-1.png)

## Training-Time Test

![EAGLE-3 training flow](assets/ttt_training.png)

Another important method introduced in the paper is TTT(Training-Time Test) where for each token position, three verifier layers are concatenated and projected into the draft state.

## Quick start

Install the dependencies:

```bash
uv sync --group dev
uv pip install vllm==0.25.1
```

Each stage can be run independently. To run the full pipeline:

```bash
uv run python -m tiny_speculators.pipeline --max-samples <num_max_samples>
```

What `pipeline.py` does:

1. prepares data
2. generates verifier hidden states
3. trains the draft
4. exports it to vLLM's checkpoint format
5. verifies that vLLM uses the draft model

### Resuming

To resume a checkpoint after epoch two and train through epoch five:

```bash
uv run python -m tiny_speculators.scripts.train_eagle3 \
  --resume checkpoints/eagle3 \
  --output checkpoints/eagle3 \
  --epochs 5
```

or use pipeline script. 

It reuses the existing data artifacts,
continues training, exports the checkpoint, and runs the demo:

```bash
uv run python -m tiny_speculators.pipeline \
  --resume checkpoints/eagle3 \
  --epochs 5
```

## Future Works

- [x] EAGLE-3
- [ ] In-depth article on training EAGLE-3 from scratch
- [ ] EAGLE-3.1
- [ ] DFlash
- [ ] DSpark
