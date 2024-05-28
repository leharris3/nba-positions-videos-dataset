# Setting

---

## Test Set

| # Clips | Duration (Seconds) | FPS |
| :---: | :---: | :---: |
| 50 | 10 | 30.0 |

## ROI Detection Config

| `ROI_STEP` | `CONF_THRESH` | `PAD` |
| :---: | :---: | :---: |
| 5 | 0.90 | 3 |

## Paddle OCR Config

| Rec Model | Step |
| :---: | :---: | 
| `PP-OCRv4` | 1 |

## Post Processing Config

| Post-Processing |
| :---: |
| None |

# Experiments

---

## Baseline

| Post-Processing | Time Remaining Recall | Mean Avg. Time-Remaining Diff |
| :---: | :---: |
| None | 0.985 | 0.269 |
| Extend | 1.000 | 0.273 |
| Extend + Interpoltate | **1.000** | **0.067** | 

