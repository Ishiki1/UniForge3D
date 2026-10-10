# UniForge3D: Joint 3D Reconstruction and Generation in a Shared Anchor-Guided Space

![UniForge3D teaser](assets/teaser.png)

UniForge3D generates complete 3D assets from sparse, unposed multi-view images. It jointly models reconstruction and generation in a shared anchor-guided space, then uses structure-guided view selection to preserve observed details. 

## Installation

Run from the repository root:

```bash
conda create -n uniforge3d --override-channels -c conda-forge python=3.10 -y
conda activate uniforge3d

python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install setuptools==80.9.0 wheel ninja cmake==3.31.6 packaging \
    transformers==4.57.3 peft==0.17.1 "huggingface-hub>=0.36,<1" \
    "timm>=1.0,<2" "kornia>=0.8,<0.9" "numpy>=2.1,<2.3" \
    "Pillow>=10.4,<12" "safetensors>=0.7,<1" trimesh==4.6.3 \
    "opencv-python-headless>=4.10,<5" "easydict>=1.13,<2" \
    "tqdm>=4.66,<5" "scipy>=1.13,<2"
```

### CUDA extensions

```bash
python -m pip install flash-attn==2.7.3 --no-build-isolation

git clone --recursive https://github.com/microsoft/TRELLIS.2.git .deps/TRELLIS.2
git -C .deps/TRELLIS.2 checkout 75fbf0183001ed9876c8dbb35de6b68552ee08bd
git -C .deps/TRELLIS.2 submodule update --init --recursive

git clone --recursive https://github.com/JeffreyXiang/CuMesh.git .deps/CuMesh
git -C .deps/CuMesh checkout 12289e1062f0603f2f0d0771b02e1395d247f26f
git -C .deps/CuMesh submodule update --init --recursive

git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git .deps/FlexGEMM
git -C .deps/FlexGEMM checkout 6dd94a859c26ee8246888502eada3dd8ad85532e
git -C .deps/FlexGEMM submodule update --init --recursive

git clone --branch v0.4.0 https://github.com/NVlabs/nvdiffrast.git .deps/nvdiffrast

python -m pip install ./.deps/nvdiffrast --no-build-isolation
python -m pip install ./.deps/CuMesh --no-build-isolation
python -m pip install ./.deps/FlexGEMM --no-build-isolation
python -m pip install ./.deps/TRELLIS.2/o-voxel --no-build-isolation

```

## Inference

Place the views of each object directly in a separate folder under `assets/`:

```text
assets/
├── case01/
│   ├── 001.png
│   ├── 002.png
│   ├── 003.png
│   └── 004.png
└── case02/
    └── ...
```

Run all cases using the pretrained model:

```bash
python infer.py
```

Results are saved to `outputs/<case>/prediction.glb`.
Use `--model` for another Hub repository or local model directory, or `--config` for a local JSON configuration.
Local checkpoint paths in `--config` are relative to the repository's `weights/` directory unless `weights_root` is provided through the Python API.


## Citation

```bibtex
@misc{chen2026uniforge3d,
  title={UniForge3D: Joint 3D Reconstruction and Generation in a Shared Anchor-Guided Space},
  author={Chen, Wenjue and Chen, Wenyue and Zheng, Heliang and Jia, Rongfei and Liu, Yuan and Li, Meng},
  year={2026}
}
```

## Acknowledgements

Built on [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) and [VGGT](https://github.com/facebookresearch/vggt).
