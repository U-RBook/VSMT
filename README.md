# VSMT

PyTorch implementation of "Virtual Immunohistochemistry Staining with Dual-Aligned Multi-Task Feature Guidance".

## Installation

```bash
conda env create -f environment.yml
conda activate vsmt
```

## Pretrained models

Download the pretrained generators from [Baidu Netdisk](https://pan.baidu.com/s/1GT8wQ1OeVJyyQ-0g3gVjDg?pwd=72d8).
Extraction code: `72d8`.

## Testing

Prepare paired test images with identical filenames:

```text
DATA_ROOT/
└── TrainValAB/
    ├── trainA/
    ├── trainB/
    ├── valA/
    └── valB/
```

Run inference with the experiment launcher:

```bash
python -m experiments vsmt_launcher test 0
```

Results are saved to `results/<model_name>/test_latest`.


## Training

Train the HE and IHC auxiliary models before training VSMT.

1. Train the HE and IHC reconstruction models:

```bash
python -m experiments rec_launcher train 0
```

2. Train the HE and IHC classification models:

```bash
python -m experiments cls_launcher train 0
```

For MIST, use the classification auxiliary models trained on BCI.

3. Set `he_rec`, `ihc_rec`, `he_cls`, and `ihc_cls` in
[`experiments/vsmt_launcher.py`](experiments/vsmt_launcher.py), then train VSMT:

```bash
python -m experiments vsmt_launcher train 0
```

## Citation

```bibtex
@InProceedings{Xie_2026_CVPR,
    author    = {Xie, Shigeng and Xu, Hongming and Jiang, Guiyang and Rossi, Tuomo and K\"arkk\"ainen, Tommi and Cong, Fengyu},
    title     = {Virtual Immunohistochemistry Staining with Dual-Aligned Multi-Task Feature Guidance},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {35311-35320}
}
```



## Acknowledgments

This source code is inspired by [CUT](https://github.com/taesungp/contrastive-unpaired-translation).

## Contact

[shigeng.s.xie@jyu.fi](mailto:shigeng.s.xie@jyu.fi)
