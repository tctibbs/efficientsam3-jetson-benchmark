"""Command-line interface for efficientsam3-jetson-benchmark."""

import argparse
import json
from pathlib import Path

from efficientsam3_jetson_benchmark import accuracy, download, latency, saco_gold

# First COCO val2017 image, the one eval_coco.py also takes first.
DEFAULT_IMAGE = Path("data/coco/images/val2017/000000397133.jpg")


def positive_int(text: str) -> int:
    """Parse an integer of at least 1."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for all commands."""
    parser = argparse.ArgumentParser(prog="efficientsam3-jetson-benchmark")
    commands = parser.add_subparsers(dest="command", required=True)

    fetch = commands.add_parser("download", help="Download pinned, verified inputs")
    targets = fetch.add_subparsers(dest="target", required=True)
    checkpoint = targets.add_parser("checkpoint", help="Model checkpoints")
    checkpoint.add_argument("names", nargs="+", choices=sorted(download.CHECKPOINTS))
    checkpoint.add_argument("--root", type=Path, default=Path("checkpoints"))
    coco = targets.add_parser("coco", help="COCO val2017 annotations and images")
    coco.add_argument(
        "--first",
        type=positive_int,
        required=True,
        help="Number of images, in the order eval_coco.py uses",
    )
    coco.add_argument("--root", type=Path, default=Path("data/coco"))
    gold = targets.add_parser("saco-gold", help="SA-Co/Gold labels and images")
    gold.add_argument("--root", type=Path, default=Path("data/saco-gold"))

    measure = commands.add_parser("accuracy", help="Run accuracy benchmarks")
    suites = measure.add_subparsers(dest="suite", required=True)
    coco_accuracy = suites.add_parser(
        "coco", help="eval_coco.py box-prompt mIoU on the first N images"
    )
    coco_accuracy.add_argument(
        "--checkpoint",
        required=True,
        choices=sorted(
            name for name in download.CHECKPOINTS if name.startswith("efficient_sam3_")
        ),
    )
    coco_accuracy.add_argument("--first", type=positive_int, required=True)
    coco_accuracy.add_argument(
        "--checkpoints-root", type=Path, default=Path("checkpoints")
    )
    coco_accuracy.add_argument("--coco-root", type=Path, default=Path("data/coco"))
    coco_accuracy.add_argument(
        "--upstream", type=Path, default=Path("third_party/efficientsam3")
    )
    coco_accuracy.add_argument("--results", type=Path, default=Path("results"))
    gold_accuracy = suites.add_parser(
        "saco-gold", help="SA-Co/Gold text-prompt cgF1 on the first N images per domain"
    )
    gold_accuracy.add_argument(
        "--model", required=True, choices=sorted(latency.FULL_MODELS)
    )
    gold_accuracy.add_argument(
        "--precision",
        default="bf16",
        choices=latency.PRECISIONS,
        help="Autocast precision; upstream's SA-Co/Gold script uses bf16",
    )
    gold_accuracy.add_argument("--images-per-subset", type=positive_int, default=100)
    gold_accuracy.add_argument(
        "--subsets", nargs="+", choices=list(saco_gold.SUBSETS), default=None
    )
    gold_accuracy.add_argument(
        "--checkpoints-root", type=Path, default=Path("checkpoints")
    )
    gold_accuracy.add_argument("--saco-root", type=Path, default=Path("data/saco-gold"))
    gold_accuracy.add_argument(
        "--upstream", type=Path, default=Path("third_party/efficientsam3")
    )
    gold_accuracy.add_argument("--results", type=Path, default=Path("results"))

    timing = commands.add_parser("latency", help="Run latency benchmarks")
    backends = timing.add_subparsers(dest="backend", required=True)
    pytorch = backends.add_parser(
        "pytorch", help="PyTorch text-prompt pipeline, stage by stage"
    )
    pytorch.add_argument("--model", required=True, choices=sorted(latency.FULL_MODELS))
    pytorch.add_argument("--precision", default="fp32", choices=latency.PRECISIONS)
    pytorch.add_argument("--prompt", default="person")
    pytorch.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    pytorch.add_argument("--warmup", type=positive_int, default=50)
    pytorch.add_argument("--iterations", type=positive_int, default=300)
    pytorch.add_argument("--checkpoints-root", type=Path, default=Path("checkpoints"))
    pytorch.add_argument(
        "--upstream", type=Path, default=Path("third_party/efficientsam3")
    )
    pytorch.add_argument("--results", type=Path, default=Path("results"))
    return parser


def run_download(args: argparse.Namespace) -> None:
    """Download what the download subcommand asked for."""
    if args.target == "checkpoint":
        for name in args.names:
            print(download.download_checkpoint(name, args.root))
        return
    if args.target == "saco-gold":
        print(download.download_saco_gold(args.root))
        return
    annotations = download.download_coco_annotations(args.root)
    image_ids = download.first_image_ids(annotations, args.first)
    paths = download.download_coco_images(annotations, image_ids, args.root)
    print(f"{len(paths)} images in {args.root / 'images' / 'val2017'}")


def run_accuracy(args: argparse.Namespace) -> None:
    """Run the accuracy suite and print where its results went."""
    if args.suite == "saco-gold":
        run_saco_gold(args)
        return
    run = accuracy.CocoAccuracyRun(
        checkpoint=args.checkpoint,
        first=args.first,
        checkpoints_root=args.checkpoints_root,
        coco_root=args.coco_root,
        upstream=args.upstream,
    )
    path = accuracy.run_coco_accuracy(run, args.results)
    metrics = json.loads(path.read_text())["metrics"]
    print(f"mIoU {metrics['miou']:.4f} over {metrics['instances']} instances: {path}")


def run_saco_gold(args: argparse.Namespace) -> None:
    """Run the SA-Co/Gold suite and print the average cgF1."""
    run = saco_gold.SacoGoldRun(
        model=args.model,
        precision=args.precision,
        images_per_subset=args.images_per_subset,
        subsets=tuple(args.subsets or saco_gold.SUBSETS),
        saco_root=args.saco_root,
        checkpoints_root=args.checkpoints_root,
        upstream=args.upstream,
    )
    path = saco_gold.run_saco_gold(run, args.results)
    average = json.loads(path.read_text())["average"]
    print(
        f"cgF1 {100 * average['cgF1']:.2f}, IL_MCC {average['IL_MCC']:.2f}, "
        f"pmF1 {100 * average['positive_micro_F1']:.2f} averaged over "
        f"{len(run.subsets)} domains: {path}"
    )


def run_latency(args: argparse.Namespace) -> None:
    """Run the latency benchmark and print the end-to-end result."""
    run = latency.LatencyRun(
        model=args.model,
        precision=args.precision,
        prompt=args.prompt,
        image=args.image,
        warmup=args.warmup,
        iterations=args.iterations,
        checkpoints_root=args.checkpoints_root,
        upstream=args.upstream,
    )
    path = latency.run_latency(run, args.results)
    end_to_end = json.loads(path.read_text())["end_to_end_ms"]
    print(
        f"end to end p50 {end_to_end['p50_ms']:.1f} ms, "
        f"{end_to_end['fps']:.1f} FPS from the mean: {path}"
    )


def main(argv: list[str] | None = None) -> None:
    """Run the command given on the command line."""
    args = build_parser().parse_args(argv)
    commands = {
        "download": run_download,
        "accuracy": run_accuracy,
        "latency": run_latency,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
