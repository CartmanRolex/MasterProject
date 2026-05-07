# Trained Models

Index of trained policy models. Weights are stored on HuggingFace Hub under `MasterProject2026` — not in this repo.

## Models

| HuggingFace ID | Architecture | Notes |
|----------------|--------------|-------|
| `MasterProject2026/ACT-pick-orange` | ACT | 100k steps, BS 8 |
| `MasterProject2026/pick-orange-mimic` | SmolVLA | 40k steps, BS 32 |
| `MasterProject2026/Gal-pick-orange-tailedCH20` | SmolVLA | Fine-tuned on tailed subtask data; current best |

## Eval Results

All evaluation `.txt` logs and `plot.py` are consolidated in `isaac-inference/results/`.

## Loading a Model

```python
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
policy = SmolVLAPolicy.from_pretrained("MasterProject2026/Gal-pick-orange-tailedCH20")
```
