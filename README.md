<p align="center">
  <img src="templates/logo.png" alt="LiFT logo" width="300"/>
</p>

# LiFT – Live Foci Tracking

LiFT is an end-to-end pipeline for detecting and tracking DNA-damage foci (e.g. γH2AX, 53BP1) in live-cell fluorescence microscopy timelapse data. Starting from raw multi-frame TIFF stacks, it segments and tracks nuclei, crops each cell into its own image stack, corrects for motion, detects foci frame-by-frame, and tracks individual foci over time to quantify repair kinetics.

---

## Pipeline overview

<p align="center">
  <img src="templates/pipeline_schematic.png" alt="Pipeline schematic" width="750"/>
</p>

The pipeline runs in seven sequential steps:

| Step | Name | Description |
|------|------|-------------|
| 1 | **Nuclei segmentation** | Detect and segment nuclei in every frame using Cellpose-SAM or Cellpose-V3 |
| 2 | **Nuclei tracking** | Link segmented nuclei across frames to build single-cell trajectories |
| 3 | **Cell cropping** | Crop each tracked nucleus into a separate image stack |
| 4 | **Registration** | Correct for translational and rotation within each cropped cell stack |
| 5 | **Foci detection** | Detect DNA-damage foci in the registered stacks |
| 6 | **Foci tracking** | Track individual foci over time to quantify repair kinetics |

---

## Five ways to run LiFT

### 1. Interactive GUI — `LiFT_app.py`

A browser-based interface. Each step has a dedicated page where you can tune parameters, run a preview on a randomly selected cell, and then launch the full run on all data with a progress bar.

```bash
python LiFT_app.py
```

Then open [http://localhost:5000](http://localhost:5000) in your browser. Work through steps 1–6 in order. When you click **Run all** for any step, the current parameters are automatically saved to `parameters.yml` inside your data folder.

---

### 2. Jupyter notebook — `LiFT_notebook.ipynb`

The same pipeline exposed as a step-by-step notebook. Useful for exploratory analysis, custom preprocessing, or integrating the pipeline into a broader workflow. Each section contains a user-options cell where you set parameters, a preview cell to visualise results on a subset, and a run-all cell that saves parameters to `parameters.yml` and processes the full dataset.

```bash
jupyter notebook LiFT_notebook.ipynb
```

---

### 3. Command-line runner — `LiFT.py`

For this method, you can simply clone the entire repository and execute from that local directory. 

Reads the `parameters.yml` written by the GUI or notebook and re-runs the full pipeline on a new data folder with identical settings. More experienced users can also write a custom `parameters.yml` with their desired settings. Intended for batch reproduction and HPC submission.

```bash
# Run all steps
python LiFT.py data/

# Run specific steps only
python LiFT.py data/ --steps 1 2 5 6
```

Programmatic use:

```python
import LiFT

LiFT.run("data/")
# or specific steps only by adding: steps=[1, 2, 5, 6]
```

---
 
### 4. Python Package



LiFT is published on [PyPI](https://pypi.org/project/lift-foci/) as `lift-foci` — install it and it's available in your codebase as `lift`.



```bash

pip install lift-foci



# with an optional extra, e.g. GPU segmentation

pip install "lift-foci[cp-sam]"

```



Confirm it installed correctly:



```bash

lift info

```



Every pipeline step is available as a Python function. Parameters can come from `parameters.yml` or be passed explicitly — explicit arguments always win over config.

 

```python

import lift

 

# run the full pipeline from config

lift.run("data/experiment_name")

lift.run("data/experiment_name", steps=[1, 2, 5, 6])

 

# or call individual steps

lift.segment("data/experiment_name")

lift.segment("data/experiment_name", method="cellpose_v3", diameter=120, min_area=500)

 

lift.track_nuclei("data/experiment_name")

lift.track_nuclei("data/experiment_name", method="IOU", min_length=15)

 

lift.crop("data/experiment_name", margin=50)

 

lift.register("data/experiment_name")

lift.register("data/experiment_name", method="stackreg")

 

lift.detect("data/experiment_name")

lift.detect("data/experiment_name", method="TopHat", threshold=36, sigma=0.6, radius=3.0)

 

lift.track_foci("data/experiment_name")

lift.track_foci("data/experiment_name", method="GNN", max_distance=8.0, gap_closing=3)

```

 

---



## CLI reference

 

```

lift info                                         # show installed optional packages

lift init [data_path]                             # generate parameters.yml (default: cwd)

lift config [data_path] [key.path=value ...]      # interactive editor or set values directly

lift run <data_path> [--steps N ...]              # run pipeline

lift <data_path> [--steps N ...]                  # shorthand for lift run

```

 

### `lift init`

 

Probes your environment for installed optional packages and writes a `parameters.yml` with the best available defaults, also checks currently installed packages and inserts these as options:

 

```bash

lift init                          # writes to current directory

lift init data/experiment_name     # writes to specified path

```

 

### `lift config`

 

With no arguments, opens an interactive editor that walks through every parameter step by step, showing the current value and available options. Press Enter to keep a value unchanged, or type a new one. Ctrl+C saves what has been changed so far and exits.

 

```bash

lift config                        # interactive editor in current directory

lift config data/experiment_name   # interactive editor at path

```

 

To set values directly without the interactive editor:

 

```bash

lift config step5_detection.threshold=40

lift config step5_detection.method=TopHat step5_detection.params.radius=3.0

lift config step1_segmentation.segmentation.min_area=500

lift config step4_registration.preprocessing=null

lift config data/experiment_name step5_detection.threshold=40

```

 

With no `key=value` arguments, prints the current config.

 

---



### 5. Docker Image



## Docker

 

LiFT ships a single `Dockerfile` that accepts two build arguments, so you never need multiple Dockerfiles.

 

### Build arguments

 

| Argument | Description | Default |

|----------|-------------|---------|

| `BASE_IMAGE` | Base image — use `python:3.11-slim` for CPU or `nvidia/cuda:12.1.0-runtime-ubuntu22.04` for GPU | `python:3.11-slim` |

| `EXTRAS` | Comma-separated pip extras to install | _(none)_ |

 

### Pull from Docker Hub



Published images — no build required. Replace `<dockerhub-user>` with the actual namespace once you know it:



```bash

docker pull <dockerhub-user>/lift:cpu    # cellpose-v3, no GPU needed

docker pull <dockerhub-user>/lift:gpu    # cellpose-SAM, requires NVIDIA container runtime

docker pull <dockerhub-user>/lift:full   # cpu-sam + trackastra + spotiflow



# pin to a specific released version instead of the rolling tag

docker pull <dockerhub-user>/lift:0.1.0-cpu

```



`elastix`/`NGMA`/`NND` are Windows-only and experimental — they are not included in any published image (see Platform support note below). Skip to **Running** below, using `<dockerhub-user>/lift:cpu` (etc.) in place of `lift-cpu`.



### Building an image locally



Only needed for a custom `EXTRAS` combination not covered by the published tags above.

 

```bash

# base only — no segmentation backend

docker build -t lift .

 

# CPU: cellpose-v3, no GPU needed

docker build --build-arg EXTRAS="cp-v3" -t lift-cpu .

 

# GPU: cellpose-SAM, requires NVIDIA container runtime

docker build \

  --build-arg BASE_IMAGE=nvidia/cuda:12.1.0-runtime-ubuntu22.04 \

  --build-arg EXTRAS="cp-sam" \

  -t lift-gpu .

 

# full: everything available in Docker (elastix/NGMA/NND are Windows-only, excluded — see Platform support above)

docker build \

  --build-arg BASE_IMAGE=nvidia/cuda:12.1.0-runtime-ubuntu22.04 \

  --build-arg EXTRAS="cp-sam,trackastra,spotiflow" \

  -t lift-full .

 

# custom combination

docker build --build-arg EXTRAS="cp-v3,trackastra" -t lift-custom .

```

 

### Running

 

Mount your data folder into `/data` and pass the same commands as the CLI. Examples below use the locally built tags (`lift-cpu`, `lift-gpu`) — swap in `<dockerhub-user>/lift:cpu` etc. if you pulled from Docker Hub instead, the usage is identical:

 

```bash

# generate config

docker run --rm -v $(pwd)/data:/data lift-cpu init /data/experiment_name

 

# run all steps

docker run --rm -v $(pwd)/data:/data lift-cpu run /data/experiment_name

 

# run specific steps

docker run --rm -v $(pwd)/data:/data lift-cpu run /data/experiment_name --steps 5 6

 

# GPU run

docker run --rm --gpus all -v $(pwd)/data:/data lift-gpu run /data/experiment_name

 

# check installed packages

docker run --rm lift-cpu info

```

 

On Windows use `%cd%` instead of `$(pwd)`:

 

```bash

docker run --rm -v %cd%/data:/data lift-cpu run /data/experiment_name

```

 

### Using docker compose

 

The `docker-compose.yml` defines three pre-configured profiles (`cpu`, `gpu`, `full`) and a `custom` profile driven by environment variables:

 

```bash

# build a profile

docker compose --profile cpu build

docker compose --profile gpu build

 

# run

docker compose --profile cpu  run cpu  run /data/experiment_name

docker compose --profile gpu  run gpu  run /data/experiment_name

docker compose --profile full run full run /data/experiment_name --steps 5 6

 

# custom combination via environment variables

BASE_IMAGE=python:3.11-slim EXTRAS=cp-v3,trackastra docker compose --profile custom build custom

docker compose --profile custom run custom run /data/experiment_name

```

 

### Notes

 

**NVIDIA runtime** is required for GPU images. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) and ensure `nvidia-smi` works inside a container before using the GPU profiles.

 
**Image sizes**: the GPU images are large (~5–8 GB) due to the CUDA base and torch. The CPU image with `cp-v3` is around 3 GB. Build once and reuse.

---

## Parameter reproducibility

Every time you click **Run all** in the GUI or execute the run-all cell in the notebook, the full parameter set is written to `<data_folder>/parameters.yml`. This file records all settings for every step that has been run, grouped by step:

```yaml
step1_segmentation:
  segmentation:
    method: cellpose_sam
    flow_threshold: 0.0
    cellprob_threshold: -0.5
    min_area: 1000
    cellpose_sam:
      scale_factor: 0.5
    cellpose_v3:
      diameter: 140.0
  preprocessing:
    method: None
    wavelet_filtering:
      scales: 5
      w_factor: 1.5
      start_scale: 2
    contrast_adjuster:
      sigma: 3.0
      c_factor: 6.0

:
:
:

step5_detection:
  method: TopHat
  threshold: 36.0
  return_segmentation: false
  params:
    sigma: 0.6
    radius: 6.0
# ... steps 2–4 and 6 follow the same pattern
```

Experienced users can also write a minimal `parameters.yml` by hand — any sub-keys that are absent fall back to function defaults, so only the settings that differ from defaults need to be specified.

---

## Supported algorithms

### Nuclei segmentation (Step 1)
| Method | Notes |
|--------|-------|
| `cellpose_sam` | Cellpose with SAM backbone; scale-factor controls effective nucleus size |
| `cellpose_v3` | Cellpose V3 with diameter parameter |

Optional preprocessing before segmentation to transform the foci signal into a signal resembling a nuclear (DAPI like) staining:
- `wavelet_filtering` — multi-scale wavelet coefficient thresholding
- `contrast_adjuster` — Gaussian-based contrast adjustment

### Nuclei tracking (Step 2)
| Method | Notes |
|--------|-------|
| `IOU` | Intersection-over-union based linking |
| `NND` | Greedy Nearest Neighbour Diffusion |
| `trackastra` | deep-learning tracker (Trackastra) |

### Registration (Step 4)
| Method | Notes |
|--------|-------|
| `stackreg` | StackReg rigid translation (PyStackReg) |
| `elastix` | Elastix; supports MSE, MI, and NCC loss functions |

Optional preprocessing for registration: `wavelet_denoise`, `threshold`, `DOG_filter`.

### Foci detection (Step 5)
| Method | Notes |
|--------|-------|
| `TopHat` | top-hat morphological filter |
| `LOG` | Laplacian of Gaussian |
| `Hessian` | Hessian-based blob detector |
| `Wavelets` | Isotropic undecimated wavelet transform with thresholding of the coefficients |
| `HDome` | H-dome transform |
| `HDome-smal` | H-dome with LOG pre-filter |
| `MPHD` | Maximum Possible Height Dome |
| `Spotiflow` | deep-learning spot detector (Spotiflow) |

### Foci tracking (Step 6)
| Method | Notes |
|--------|-------|
| `GNN` | Global Nearest Neighbour |
| `NGMA` | Non-iterative Greedy Multi-frame Assignment |
| `trackastra` | deep-learning tracker |

---

## Expected data structure

LiFT expects input data organised as:

```
data_folder/
  condition/
    experiment_date/
      Pos001/
        raw/ 
          0000.tif
          0001.tif   
          0002.tif  
          :       
      Pos002/
        raw/
      ...
  parameters.yml          ← written automatically; can be pre-populated
```

Intermediate and final outputs are written alongside the input inside each `Pos*/results/` folder.

---

## Installation

> **Note:** Detailed installation instructions will be added here.

---

## Citation

If you use LiFT in your research, please cite:

> *Citation will be added here upon publication.*
