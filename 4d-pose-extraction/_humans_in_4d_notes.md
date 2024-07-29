### Overview
- Human mesh recovery (recovery is a weird word)
- Track from monocular video
- 3D poses (meshes) + time = 4D

### Prev Work
- PHALP (3d tracking)

### Method
1. for img in vid : results <- vit(img)
    - img -> vit -> (pose, person shape, camera[camera calib]) as a single learn able embed
    - pose  : dim(24, 3, 3)
    - shape : dim(10)
    - mesh  : pose + shape
    - camera calib
    - human mesh (3d pose) = f(image, [pose, shape, camera calib])
    - 3 parameters used to construct 3d pose, given image predict params
2. track people w/ PHALP (association)
    - tracking
    - **absolutely no need for us to use PHALP, we already have tracklets**
    - **write a simple script that iterates over bbxs and predicts 3D poses**