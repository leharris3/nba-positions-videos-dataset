# Setting
---

### Test Set

| # Clips | Duration (Sec) | FPS | Resolution |
| :---: | :---: | :---: | :---: |
| 50 | 10 | 30 | 1280x720 |

### Dependecies

| ROI Det Library | Step Size | `CONF_THRESH` |
| :---: | :---: | :---: |
| YOLO-V8 | 5 | 0.1 |

#### OCR Config

| Step Size | Threshold |
| :---: | :---: |
| 1 | **TODO** |

---

# Experiments

### OCR Backbone

| ROI Pad | Time Remaining Avg. Mean Abs Error |
| :---: | :---: | 
| `PP-OCRv4` | 3.018 |
| `trocr-small-printed` | 3.687 |
| `trocr-large-printed` | 3.557 |
| `trocr-base-stage1` | 3.550 |
| `MiniCPM-V-2` | --- |