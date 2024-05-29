## Setting

#### Test Set

| # Videos | Duration (Seconds) | FPS |
| :---: | :---: | :---: |
| 50 | 10 | 30 |

#### ROI Det Parameters

| Step Size |
| :---: |
| 5 |

#### OCR Parameters

| Step Size | Image Height |
| :---: | :---: | 
| 1 | 100 |

## Experiments

#### OCR Step Size

| OCR Step Size | Total Time (Seconds) | Time Per Video (Seconds) | Mean Err (Seconds) | Mean Recall |
| :---: | :---: | :---: | :---: | :---: | 
| 1 | 127.0 | 2.54 | 0.069 | 1.000 |
| 2 | 100.0 | 2.00 | 0.081 | 0.999 |
| **3** | **89.0** | **1.78** | **0.081** | **0.998** |
| 5 | 80.0 | 1.60 | 0.102 | 0.997 |

#### ROI Step Size

| ROI Step Size | Total Time (Seconds) | Time Per Video (Seconds) | Mean Err (Seconds) | Mean Recall |
| :---: | :---: | :---: | :---: | :---: | 
| 1 | 103.0 | 2.06 | 0.081 | 0.998 | 
| 5 | 89.0 | 1.78 | 0.081 | 0.998 |
| **15** | **88.0** | **1.76** | **0.081** | **0.998** |
| 30 | 88.0 | 1.76 | 0.081 | 0.998 |

#### Image Size

| Image Resize Height | Total Time (Seconds) | Mean Err (Seconds) | Mean Recall |
| :---: | :---: | :---: | :---: |
| None | 85.0 | 0.0828 | 0.998|
| 100 | 88.0| 0.0810 | 0.998 |
| 80 | 86.0 | 0.0819 | 0.998 |
| **50** | **83.0** | **0.0815** | **0.998** |
| 30 | 84.0 | 0.0826 | 0.998 |

#### Process Pool vs. Thread Pool

| Concurrent Type | Total Time (Seconds) | Mean Err (Seconds) | Mean Recall |
| :---: | :---: | :---: | :---: |
| Process Pool | 83.0 | 0.0815 | 0.998 |
| **Thread Pool** | **45.0** | **0.0815** | **0.998** |

#### Queue Blocking

| Queue Blocking Enabled | Total Time (Seconds) | Mean Err (Seconds) | Mean Recall |
| :---: | :---: | :---: | :---: |
| True | 45.0 | 0.0815 | 0.998 |
| **False** | **41.0** | **0.0815** | **0.998** |

