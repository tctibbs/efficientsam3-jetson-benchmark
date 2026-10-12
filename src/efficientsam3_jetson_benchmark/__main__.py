"""Command-line interface for efficientsam3-jetson-benchmark."""

import argparse
import json
from pathlib import Path

from efficientsam3_jetson_benchmark import accuracy, download


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

    measure = commands.add_parser("accuracy", help="Run accuracy benchmarks")
    suites = measure.add_subparsers(dest="suite", required=True)
    coco_accuracy = suites.add_parser(
        "coco", help="Tier A: eval_coco.py box-prompt mIoU on the first N images"
    )
    coco_accuracy.add_argument(
        "--checkpoint",
        required=True,
        choices=sorted(name for name in download.CHECKPOINTS if name != "sam3"),
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
    return parser


def run_download(args: argparse.Namespace) -> None:
    """Download what the download subcommand asked for."""
    if args.target == "checkpoint":
        for name in args.names:
            print(download.download_checkpoint(name, args.root))
        return
    annotations = download.download_coco_annotations(args.root)
    image_ids = download.first_image_ids(annotations, args.first)
    paths = download.download_coco_images(annotations, image_ids, args.root)
    print(f"{len(paths)} images in {args.root / 'images' / 'val2017'}")


def run_accuracy(args: argparse.Namespace) -> None:
    """Run the accuracy suite and print where its results went."""
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


def main(argv: list[str] | None = None) -> None:
    """Run the command given on the command line."""
    args = build_parser().parse_args(argv)
    if args.command == "download":
        run_download(args)
    else:
        run_accuracy(args)


if __name__ == "__main__":
    main()
