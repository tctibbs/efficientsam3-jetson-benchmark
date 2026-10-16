# Speed (PyTorch)

Each stage-3 model was given one image (COCO `000000397133.jpg`) and one text prompt ("person") through EfficientSAM3's `Sam3Processor`. The clocks were locked. After 50 warm-up frames, 300 frames were timed with CUDA events. Power is the Jetson module's total input power (VDD_IN) as reported by tegrastats.

![Frames per second by precision and time per step](speed.svg)

| Model | Precision | FPS | Median ms | p99 ms | Power W | J per frame |
|---|---|--:|--:|--:|--:|--:|
| EV-M | fp32 | 1.62 | 616 | 622 | 17.8 | 11.0 |
| EV-M | fp16 | **2.74** | **365** | 367 | 15.9 | **5.8** |
| EV-M | bf16 | 2.62 | 381 | 388 | 15.4 | 5.9 |
| RV-M | fp32 | 1.55 | 643 | 650 | 17.8 | 11.5 |
| RV-M | fp16 | 2.63 | 379 | 385 | 16.0 | 6.1 |
| RV-M | bf16 | 2.48 | 403 | 410 | 15.4 | 6.2 |
| TV-M | fp32 | 1.45 | 689 | 693 | 18.1 | 12.5 |
| TV-M | fp16 | 2.37 | 422 | 425 | 16.3 | 6.9 |
| TV-M | bf16 | 2.29 | 436 | 445 | 15.9 | 6.9 |

The decoder step is EfficientSAM3's `forward_grounding` function. It combines the image and text features, finds the objects, and predicts their masks. It runs once for each text prompt. The text encoder takes only 1.5 ms longer for 16 prompts than for one.

Each result file also records the p90 latency, the time of every step, GPU load, peak memory, peak temperature, and what the model found.
