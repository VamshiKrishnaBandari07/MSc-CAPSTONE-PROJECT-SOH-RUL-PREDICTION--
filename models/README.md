# Models

| File | Description |
|:---|:---|
| `../model_paper.py` | Paper hybrid **CNN–TCN–LSTM–Attention** (~0.39M params, SOH head) |

Build via:

```python
from model_paper import build_paper_model
model = build_paper_model(seq_len=300)
```
