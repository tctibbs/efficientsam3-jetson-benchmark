# EfficientSAM3 Jetson benchmark

How fast and how accurate [EfficientSAM3](https://github.com/SimonZeng7108/efficientsam3) is on an NVIDIA Jetson Orin Nano Super. The benchmarks run the published EfficientSAM3 code without changes.

This project isn't affiliated with the EfficientSAM3 authors, Meta, or NVIDIA.

## Examples

These examples come from TV-M, one of the EfficientSAM3 models, running on the Jetson. It was asked to find COCO categories by name in 300 photos. The rows go from what it does well at the top to what it gets wrong at the bottom.

![EfficientSAM3 on 25 COCO photos, from what it does well to what it does badly](results/examples/coco/gallery.jpg)

TV-M is good at large objects and found 69% of them. It can also pick out several of the same thing in one photo.

Small objects are much harder. It found just 6% of those under 32 by 32 pixels.

When asked for something that isn't in the photo, it usually marks nothing. About 1 time in 20 it marks something anyway. Most of those marks land on another real object.

## Models

EfficientSAM3 makes Meta's SAM 3 smaller by swapping its large encoders for small ones. Its checkpoints come from different training stages. Two of those stages are tested here. Stage 1 replaces only the image encoder. Stage 3 also replaces the text encoder and fine-tunes the result.

| Checkpoints | Image encoder | Text encoder | Tested here |
|---|---|---|---|
| Stage 1: ES-EV, ES-RV, ES-TV in sizes S, M, L | EfficientViT-B0 to B2, RepViT-M0.9 to M2.3, or TinyViT-5M to 21M | SAM 3's original | Box prompts |
| Stage 3: EV-M, RV-M, TV-M | EfficientViT-B1, RepViT-M1.1, or TinyViT-11M | MobileCLIP-S0 | Speed and text prompts |

## Results

All results were measured on a Jetson Orin Nano Super Developer Kit (8 GB) in MAXN_SUPER power mode. The software was JetPack 6 (L4T 36.4.4), PyTorch 2.8.0, and CUDA 12.6.

### Speed

How long each stage-3 model takes to find objects in one image from one text prompt. The left chart shows frames per second at each precision. The right chart shows how long each step takes in fp16.

![Frames per second by precision and time per step](results/latency/pytorch/speed.svg)

[Full results](results/latency/pytorch/)

### Box-prompt accuracy

How closely each stage-1 model outlines the objects in COCO val2017 when given each object's bounding box. The score is mean IoU. Models with larger image encoders sit further to the right.

![COCO mIoU by image encoder size](results/accuracy/coco/miou.svg)

[Full results](results/accuracy/coco/)

### Text-prompt accuracy

How well each stage-3 model finds every object matching a short phrase like "gas grill". The test uses the first 100 images from each of the seven subsets in Meta's SA-Co/Gold benchmark. That is 7.6% of the benchmark's image-phrase pairs. The score is cgF1, the benchmark's official metric. It is high only when a model correctly says whether the phrase appears in the image and also outlines the matching objects well. 100 would be perfect.

The best model, TV-M, scores 14.9. Meta reports 54.1 for SAM 3 on the full benchmark. TV-M notices the phrase in about half of the images that contain it. It wrongly reports the phrase in 4% of the images that don't contain it.

![SA-Co/Gold cgF1 by subset](results/accuracy/saco-gold/cgf1.svg)

[Full results](results/accuracy/saco-gold/)

## Running the benchmarks

You'll need JetPack 6 with CUDA 12.6, [uv](https://docs.astral.sh/uv/), up to 22 GB of disk space, and about 5 GB of free memory.

```bash
git clone --recurse-submodules https://github.com/tctibbs/efficientsam3-jetson-benchmark.git
cd efficientsam3-jetson-benchmark
uv sync
sudo nvpmodel -m 2 && sudo jetson_clocks  # MAXN_SUPER, clocks locked

uv run efficientsam3-jetson-benchmark download checkpoint efficientsam3_tinyvit efficient_sam3_tinyvit_m
uv run efficientsam3-jetson-benchmark download coco --first 5000
uv run efficientsam3-jetson-benchmark download saco-gold

uv run efficientsam3-jetson-benchmark latency pytorch --model efficientsam3_tinyvit --precision fp16
uv run efficientsam3-jetson-benchmark accuracy coco --checkpoint efficient_sam3_tinyvit_m --first 5000
uv run efficientsam3-jetson-benchmark accuracy saco-gold --model efficientsam3_tinyvit
uv run efficientsam3-jetson-benchmark examples coco
uv run efficientsam3-jetson-benchmark figures
```

Each run saves its results as a JSON file under `results/`. The `figures` command redraws the charts from these files. Downloaded checkpoints and annotations are checked against known SHA-256 hashes.

The EfficientSAM3 code is a git submodule pinned to commit `bd0936c`. Each result file records the code versions, software versions, Jetson settings, and checksums of the model and data used.

## Development

```bash
uv run pytest && uv run ruff check . && uv run ty check
```

## License

This repository is licensed under the [MIT License](LICENSE).

This project builds upon [EfficientSAM3](https://github.com/SimonZeng7108/efficientsam3), [SAM 3](https://github.com/facebookresearch/sam3), [COCO](https://cocodataset.org/), and [SA-Co/Gold](https://huggingface.co/datasets/facebook/SACo-Gold). Please refer to their respective licenses for usage terms.

The example photos are from COCO val2017 and are used under the [Creative Commons Attribution 2.0 license](http://creativecommons.org/licenses/by/2.0/). See the [photo credits](results/examples/coco/CREDITS.md).

## Acknowledgments

Special thanks to the amazing work by Chengxi Simon Zeng and the EfficientSAM3 team for making SAM 3 small enough to run on edge hardware. This benchmark relies on their released models and evaluation code. Thanks also to Meta for SAM 3 and the SA-Co/Gold benchmark. If you use EfficientSAM3 in your work, please cite the [EfficientSAM3 paper](https://arxiv.org/abs/2511.15833).
