# Box-prompt accuracy (COCO)

These results come from EfficientSAM3's unmodified `eval/eval_coco.py` evaluation script. It gives each stage-1 model the bounding box of every object in COCO val2017 and compares the predicted mask with the true outline. The box goes through SAM 3's interactive mask decoder. The text encoder isn't used. The script runs in bfloat16 and skips crowd regions. The score is the mean IoU over all 36,335 objects.

![COCO mIoU by image encoder size](miou.svg)

| Model | Image encoder | Parameters | mIoU | Run time |
|---|---|--:|--:|--:|
| ES-EV-S | EfficientViT-B0 | 10.3M | 0.6159 | 31 min |
| ES-EV-M | EfficientViT-B1 | 14.3M | 0.6483 | 34 min |
| ES-EV-L | EfficientViT-B2 | 24.8M | **0.6630** | 37 min |
| ES-RV-S | RepViT-M0.9 | 14.6M | 0.6478 | 33 min |
| ES-RV-M | RepViT-M1.1 | 17.7M | 0.6526 | 35 min |
| ES-RV-L | RepViT-M2.3 | 32.5M | 0.6555 | 48 min |
| ES-TV-S | TinyViT-5M | 14.8M | 0.6553 | 36 min |
| ES-TV-M | TinyViT-11M | 20.4M | 0.6547 | 38 min |
| ES-TV-L | TinyViT-21M | 30.7M | **0.6630** | 43 min |

The parameter counts come from the checkpoint files. They include the image encoder's backbone and the projection layers that connect it to the rest of SAM 3. Run time includes loading the model and scoring.

Each result file also records the IoU of every object.
