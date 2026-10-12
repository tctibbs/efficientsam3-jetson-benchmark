"""Command-line interface for efficientsam3-jetson-benchmark."""

import argparse
from pathlib import Path

from efficientsam3_jetson_benchmark import download


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
        type=int,
        required=True,
        help="Number of images, in the order eval_coco.py uses",
    )
    coco.add_argument("--root", type=Path, default=Path("data/coco"))
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the command given on the command line."""
    args = build_parser().parse_args(argv)
    if args.target == "checkpoint":
        for name in args.names:
            print(download.download_checkpoint(name, args.root))
    else:
        annotations = download.download_coco_annotations(args.root)
        image_ids = download.first_image_ids(annotations, args.first)
        paths = download.download_coco_images(annotations, image_ids, args.root)
        print(f"{len(paths)} images in {args.root / 'images' / 'val2017'}")


if __name__ == "__main__":
    main()
