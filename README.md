# **Automated Basketball Shot-Quality Analysis with Video Classification Models**

**Overview**: Can a model learn the features that define a good basketball shot from video alone? We seek to answer this question by training a model to predict the outcome of basketball shot clips from pre-release context alone.

**Project Goals**:
- [x] Build a temporal-grounding pipeline for basketball broadcast footage.
- [ ] Build a shot-extraction pipeline.
  - Input: video + data (timestamps + game logs)
  - Output: shot clips (5-10s clips of a basketball shot attempt) + binary label (0 = miss | 1 = make)
    - Too much context is fine, but not enough is bad
- [ ] Shot segmentation pipeline. Split a shot clip into [pre-release | post-release | crop] categories. Add annotations.
- [ ] Train a model to predict the outcomes of annotated shot clips by providing a confidence score in range [0-1].
