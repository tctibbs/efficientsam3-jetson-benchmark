"""Draw each benchmark's figure from its result files."""

import json
from pathlib import Path
from typing import Any

MODELS = {
    "efficientsam3_efficientvit": "EV-M",
    "efficientsam3_repvit": "RV-M",
    "efficientsam3_tinyvit": "TV-M",
}
PRECISIONS = ("fp32", "fp16", "bf16")
STEPS = {
    "preprocess": "Prepare image",
    "image_encoder": "Image encoder",
    "text_encoder": "Text encoder",
    "decoder": "Decoder",
    "postprocess": "Masks and boxes",
}
ENCODERS = {"efficientvit": "EfficientViT", "repvit": "RepViT", "tinyvit": "TinyViT"}
SIZES = ("s", "m", "l")
# Millions of parameters in each stage-1 checkpoint's image encoder, backbone plus
# projection head, counted from the checkpoint files.
ENCODER_PARAMETERS = {
    ("efficientvit", "s"): 10.254,
    ("efficientvit", "m"): 14.346,
    ("efficientvit", "l"): 24.810,
    ("repvit", "s"): 14.551,
    ("repvit", "m"): 17.739,
    ("repvit", "l"): 32.500,
    ("tinyvit", "s"): 14.839,
    ("tinyvit", "m"): 20.446,
    ("tinyvit", "l"): 30.650,
}
# Where each family's point labels sit, so labels of nearby points don't collide.
LABEL_POSITIONS = {
    "efficientvit": ((-6, 4), "right"),
    "repvit": ((6, -10), "left"),
    "tinyvit": ((6, 4), "left"),
}
# Upstream README's domain order.
DOMAINS = {
    "metaclip": "MetaCLIP captions",
    "sa1b": "SA-1B captions",
    "crowded": "Crowded",
    "fg_food": "Food and drink",
    "fg_sports_equipment": "Sports equipment",
    "attributes": "Attributes",
    "wiki_common": "Wiki common",
}
# One colour per encoder family, shared by its stage-1 checkpoints and full model.
FAMILY_COLORS = {
    "efficientvit": "tab:blue",
    "repvit": "tab:orange",
    "tinyvit": "tab:green",
}
PRECISION_COLORS = {"fp32": "#c4c4c4", "fp16": "#3b3b3b", "bf16": "#8a8a8a"}
# Muted steps, with the decoder highlighted.
STEP_COLORS = {
    "preprocess": "#e8e8e8",
    "image_encoder": "#8fa9c4",
    "text_encoder": "#c9b8de",
    "decoder": "tab:red",
    "postprocess": "#5c5c5c",
}
STYLE = {
    "svg.hashsalt": "efficientsam3-jetson-benchmark",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def load_records(folder: Path) -> list[dict[str, Any]]:
    """Return every result record in a folder, oldest first."""
    return [json.loads(path.read_text()) for path in sorted(folder.glob("*.json"))]


def save(figure: Any, path: Path) -> Path:
    """Write a figure as SVG without a timestamp, so redrawing it changes nothing."""
    figure.savefig(path, format="svg", metadata={"Date": None}, bbox_inches="tight")
    return path


def plot_speed(records: list[dict[str, Any]], path: Path) -> Path:
    """Plot frames per second by precision, and where an fp16 frame's time goes."""
    from matplotlib.figure import Figure

    runs = {(r["model"]["name"], r["settings"]["precision"]): r for r in records}
    models = [model for model in MODELS if (model, "fp16") in runs]
    figure = Figure(figsize=(10, 3))
    fps_axes, steps_axes = figure.subplots(1, 2, width_ratios=[1, 1.5])

    width = 0.8 / len(PRECISIONS)
    for index, precision in enumerate(PRECISIONS):
        offset = (index - (len(PRECISIONS) - 1) / 2) * width
        bars = fps_axes.bar(
            [position + offset for position in range(len(models))],
            [runs[model, precision]["end_to_end_ms"]["fps"] for model in models],
            width,
            label=precision,
            color=PRECISION_COLORS[precision],
        )
        fps_axes.bar_label(bars, fmt="%.1f", fontsize=7)
    fps_axes.set_xticks(range(len(models)), [MODELS[model] for model in models])
    fps_axes.set_ylim(top=fps_axes.get_ylim()[1] * 1.25)
    fps_axes.set_ylabel("Frames per second")
    fps_axes.set_title("One image and one text prompt")
    fps_axes.legend(frameon=False, ncols=3, loc="upper center")

    starts = [0.0] * len(models)
    for step, label in STEPS.items():
        times = [runs[model, "fp16"]["stages_ms"][step]["p50_ms"] for model in models]
        bars = steps_axes.barh(
            range(len(models)),
            times,
            left=starts,
            label=label,
            color=STEP_COLORS[step],
            edgecolor="white",
            linewidth=0.5,
        )
        if step in ("image_encoder", "decoder"):
            steps_axes.bar_label(bars, fmt="%.0f ms", label_type="center", fontsize=7)
        starts = [start + time for start, time in zip(starts, times, strict=True)]
    steps_axes.set_yticks(range(len(models)), [MODELS[model] for model in models])
    steps_axes.invert_yaxis()
    steps_axes.set_xlabel("Median ms per frame, fp16")
    steps_axes.set_title("Where the time goes")
    # Beside its own chart, top to bottom in pipeline order.
    steps_axes.legend(frameon=False, loc="center left", bbox_to_anchor=(1, 0.5))
    return save(figure, path)


def plot_coco(records: list[dict[str, Any]], path: Path) -> Path:
    """Plot box-prompt mIoU against image encoder size, one line per family."""
    from matplotlib.figure import Figure

    miou = {}
    for record in records:
        _, _, encoder, size = record["checkpoint"]["name"].split("_")
        miou[encoder, size] = record["metrics"]["miou"]
    figure = Figure(figsize=(6, 3.4))
    axes = figure.subplots()
    for encoder, label in ENCODERS.items():
        sizes = [size for size in SIZES if (encoder, size) in miou]
        parameters = [ENCODER_PARAMETERS[encoder, size] for size in sizes]
        scores = [miou[encoder, size] for size in sizes]
        axes.plot(
            parameters, scores, marker="o", label=label, color=FAMILY_COLORS[encoder]
        )
        offset, alignment = LABEL_POSITIONS[encoder]
        for size, x, y in zip(sizes, parameters, scores, strict=True):
            axes.annotate(
                f"{size.upper()} {y:.4f}",
                (x, y),
                xytext=offset,
                textcoords="offset points",
                ha=alignment,
                fontsize=7,
            )
    axes.set_xlabel("Image encoder parameters (millions)")
    axes.set_ylabel("mIoU")
    axes.set_title("COCO val2017, box prompts")
    axes.legend(frameon=False, loc="lower right")
    return save(figure, path)


def plot_saco_gold(records: list[dict[str, Any]], path: Path) -> Path:
    """Plot text-prompt cgF1 per domain and on average, one bar per model."""
    from matplotlib.figure import Figure

    runs = {record["model"]["name"]: record for record in records}
    models = [model for model in MODELS if model in runs]
    rows = ["average", *DOMAINS]
    labels = ["Average", *DOMAINS.values()]
    figure = Figure(figsize=(6, 4.2))
    axes = figure.subplots()
    height = 0.8 / len(models)
    for index, model in enumerate(models):
        run = runs[model]
        scores = [
            run["average"]["cgF1"]
            if row == "average"
            else run["subsets"][row]["metrics"]["cgF1"]
            for row in rows
        ]
        offset = (index - (len(models) - 1) / 2) * height
        bars = axes.barh(
            [position + offset for position in range(len(rows))],
            [100 * score for score in scores],
            height,
            label=MODELS[model],
            color=FAMILY_COLORS[model.removeprefix("efficientsam3_")],
        )
        axes.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
    axes.set_yticks(range(len(rows)), labels)
    axes.invert_yaxis()
    axes.set_xlabel("cgF1 (%)")
    images = runs[models[0]]["settings"]["images_per_subset"]
    axes.set_title(f"SA-Co/Gold, text prompts, {images} images per subset")
    axes.legend(frameon=False, loc="upper right")
    return save(figure, path)


FIGURES = (
    (Path("latency", "pytorch"), "speed.svg", plot_speed),
    (Path("accuracy", "coco"), "miou.svg", plot_coco),
    (Path("accuracy", "saco-gold"), "cgf1.svg", plot_saco_gold),
)


def write_figures(results: Path) -> list[Path]:
    """Draw each benchmark's figure into its results folder, skipping empty ones."""
    import matplotlib

    paths = []
    with matplotlib.rc_context(STYLE):
        for folder, name, plot in FIGURES:
            records = load_records(results / folder)
            if records:
                paths.append(plot(records, results / folder / name))
    return paths
