# 🔄 Real-to-Sim Pipeline Documentation

## 📋 Overview

This pipeline converts real-world environments and objects into simulated representations for robotic manipulation and other applications.

---

## Part 1: Environment Scan 🌍

### 📸 2D Gaussian Splatting / Polycam

The environment scanning phase captures the spatial layout and visual appearance of the real-world scene.

**Tools:**
- 2D Gaussian Splatting for efficient scene representation
- [Polycam](https://poly.cam/) for photogrammetric reconstruction

**Process:**
1. Capture multiple images or video of the environment from various angles
2. Process the captured data to generate a 3D reconstruction
3. Export the environment model for simulation integration

---

## Part 2: Object Tracking 🎯

### 2.1 Object Scanning: Polycam 📦

Individual objects within the environment are scanned to create detailed 3D models.

**Tool:** [Polycam](https://poly.cam/)

**Process:**
1. Isolate the target object
2. Capture 360-degree coverage using Polycam
3. Generate a textured 3D mesh
4. Export the object model in a standard format (e.g., OBJ, PLY, GLTF)

### 2.2 Object Tracking: FoundationPose 🤖

[FoundationPose](https://github.com/NVlabs/FoundationPose) provides model-based 6DoF pose estimation for tracking objects in real-time.

#### 📥 Installation

```bash
git clone https://github.com/NVlabs/FoundationPose.git
cd FoundationPose
```

We followed [Env Setup Option 2: Conda (experimental)](https://github.com/NVlabs/FoundationPose?tab=readme-ov-file#env-setup-option-2-conda-experimental) for our setup.

**Common Setup Issues:**

During installation, you may encounter the following issues:
1. [Update to readme for cpp → python setup](https://github.com/NVlabs/FoundationPose/pull/185/files)
2. [Update C++ version to 17](https://github.com/NVlabs/FoundationPose/issues/35)
3. [Build without ninja](https://github.com/NVlabs/FoundationPose/issues/241)

#### 🔧 Input Requirements

FoundationPose requires four inputs:

1. **RGB-D Images:** Video frames with color and depth information
2. **Camera Intrinsics:** Focal length and principal point parameters
3. **Object Mesh(es):** 3D model(s) in OBJ format
4. **Object Mask(s):** Segmentation mask for the first frame

**Our Setup:**
- RGB-D Video + Camera Intrinsics: ZED 2i Camera
- Object Meshes: Polycam scans (see Section 2.1)
- Object Masks: SAM2 with bounding box query