# Text-prompt accuracy (SA-Co/Gold)

Each stage-3 model was asked to find every object matching a short phrase in the first 100 images of each of SA-Co/Gold's seven subsets. In total, 12,546 image-phrase pairs were scored. The phrases go through EfficientSAM3's `Sam3Processor` in bf16. SAM 3's `CGF1Evaluator` then scores the masks against the labels of all three annotators.

![SA-Co/Gold cgF1 by subset](cgf1.svg)

| Model | cgF1 | IL_MCC | pmF1 |
|---|--:|--:|--:|
| EV-M | 8.97 | 0.43 | 18.39 |
| RV-M | 12.99 | 0.51 | 23.43 |
| TV-M | **14.88** | **0.54** | **25.91** |

cgF1 is IL_MCC multiplied by pmF1. IL_MCC measures whether the model correctly says if the phrase appears in the image. pmF1 measures how well the masks match when it does. cgF1 and pmF1 are shown as percentages.

Meta's SA-Co/Gold README reports an average cgF1 of 54.06 for SAM 3 on the full benchmark. By the first annotator's labels, the Wiki common subset has only 31 pairs where the phrase appears. Its score is the least reliable of the seven. Each model took about 54 minutes to run.

Each result file records every metric for every subset.
