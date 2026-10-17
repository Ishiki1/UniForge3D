# UniForge3D: Joint 3D Reconstruction and Generation in a Shared Anchor-Guided Space

Wenjue Chen<sup>1,&#42;</sup>, Wenyue Chen<sup>1,2,&#42;</sup>, Heliang Zheng<sup>2</sup>, Rongfei Jia<sup>2</sup>, Yuan Liu<sup>3,&dagger;</sup>, Meng Li<sup>1,&dagger;</sup>

<sup>1</sup> Peking University &nbsp; <sup>2</sup> Math Magic &nbsp; <sup>3</sup> The Hong Kong University of Science and Technology

<sup>&#42;</sup> Equal contribution. <sup>&dagger;</sup> Corresponding authors.

[![Project Page](https://img.shields.io/badge/Project-Page-4285F4?style=flat&logo=googlechrome&logoColor=white)](https://xishuxishu.github.io/uniforge3d-project/)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-Models-FFD21E?style=flat&logo=huggingface&logoColor=FFD21E)](https://huggingface.co/xibushu/UniForge3D)
[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2610.15357-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2610.15357)

![UniForge3D teaser](assets/teaser.png)

UniForge3D generates complete 3D assets from sparse, unposed multi-view images. It jointly models reconstruction and generation in a shared anchor-guided space, then uses structure-guided view selection to preserve observed details. 

## Installation

```bash
git clone https://github.com/xibushu/UniForge3D.git
cd UniForge3D
bash setup.sh
conda activate uniforge3d
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



## Acknowledgements

Our code is based on these wonderful works:

- [TRELLIS.2](https://github.com/microsoft/TRELLIS.2)
- [VGGT](https://github.com/facebookresearch/vggt)

## License

Original UniForge3D source code is released under the
[MIT License](LICENSE). Third-party code, dependencies, and model
weights retain their respective licenses and are not relicensed
under this project's MIT License. Please refer to their original
license files and notices for applicable terms.

The MIT License for our source code does not apply to the released
model weights. In particular, our VGGT-derived checkpoints were
initialized from
[facebook/VGGT-1B](https://huggingface.co/facebook/VGGT-1B)
and remain subject to
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
The default background remover,
[BRIA RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0),
requires separate authorization for commercial use.

## Citation

If you find UniForge3D useful in your research, please cite our [paper](https://arxiv.org/abs/2610.15357):

```bibtex
@article{chen2026uniforge3d,
  title   = {UniForge3D: Joint 3D Reconstruction and Generation in a Shared Anchor-Guided Space},
  author  = {Chen, Wenjue and Chen, Wenyue and Zheng, Heliang and Jia, Rongfei and Liu, Yuan and Li, Meng},
  journal = {arXiv preprint arXiv:2610.15357},
  year    = {2026}
}
```

