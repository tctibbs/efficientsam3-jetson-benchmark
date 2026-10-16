"""Download pinned checkpoints, COCO val2017, and SA-Co/Gold, verifying each."""

import hashlib
import json
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CHUNK_BYTES = 1 << 20
EFFICIENTSAM3_REPO = "Simon7108528/EfficientSAM3"
EFFICIENTSAM3_REVISION = "85b05896928f974e308f889d7ccb2eefc069de98"
COCO_ANNOTATIONS_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
COCO_ANNOTATIONS_MEMBER = "annotations/instances_val2017.json"
COCO_IMAGES_URL = "http://images.cocodataset.org/val2017/"
SACO_GOLD_URL = "https://sa-co.roboflow.com/gold/all.zip"
SACO_GOLD_PARTS = ("gt-annotations", "metaclip-images", "sa1b-images")


class ChecksumError(RuntimeError):
    """A file does not match its pinned size and SHA-256."""


@dataclass(frozen=True)
class PinnedFile:
    """Expected size and SHA-256 of a file."""

    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class HubFile:
    """A file in a Hugging Face model repository at a pinned revision."""

    repo_id: str
    revision: str
    filename: str
    pin: PinnedFile


def _stage1(name: str, size_bytes: int, sha256: str) -> HubFile:
    return HubFile(
        EFFICIENTSAM3_REPO,
        EFFICIENTSAM3_REVISION,
        f"stage1_all_converted/{name}.pt",
        PinnedFile(size_bytes, sha256),
    )


def _full(name: str, size_bytes: int, sha256: str) -> HubFile:
    return HubFile(
        EFFICIENTSAM3_REPO,
        EFFICIENTSAM3_REVISION,
        f"efficientsam3_ft/{name}.pt",
        PinnedFile(size_bytes, sha256),
    )


CHECKPOINTS: dict[str, HubFile] = {
    "efficient_sam3_efficientvit_s": _stage1(
        "efficient_sam3_efficientvit_s",
        1696769051,
        "a5804a0d5c0cc685f9d686c90729f1ab6bc82380aa07a36cfb95b2895545dbcc",
    ),
    "efficient_sam3_efficientvit_m": _stage1(
        "efficient_sam3_efficientvit_m",
        1713197477,
        "309bd1e840b1a401fcd99a51da86f040cc5deae34f4f5649263d6f587731da49",
    ),
    "efficient_sam3_efficientvit_l": _stage1(
        "efficient_sam3_efficientvit_l",
        1755138822,
        "2d9330a1fc645d41df27257df11e7510085cb5fd44833c751bbe4faaaf3c30d1",
    ),
    "efficient_sam3_repvit_s": _stage1(
        "efficient_sam3_repvit_s",
        1714323359,
        "246ca1fc9fda300ef8730bc6f13a84c2eef5cc03491c3b5c3d342acb93425368",
    ),
    "efficient_sam3_repvit_m": _stage1(
        "efficient_sam3_repvit_m",
        1727097589,
        "2694a10ca7d4e470d7fd79b9c7fb0d97a902133fdb900615e1de5aff682704d0",
    ),
    "efficient_sam3_repvit_l": _stage1(
        "efficient_sam3_repvit_l",
        1786835883,
        "d4d395669026cbaa2f7d7cd964f6b5bf339507888978fe606addc954cd91a397",
    ),
    "efficient_sam3_tinyvit_s": _stage1(
        "efficient_sam3_tinyvit_s",
        1715167764,
        "d2ae4112eb3aac5bf4d3b8992f6bbf8c277189ab383b7c0144b7bfe4c7979414",
    ),
    "efficient_sam3_tinyvit_m": _stage1(
        "efficient_sam3_tinyvit_m",
        1737612329,
        "f4a044d87545a114f875e62929a88414966d931ec02377e8d9e65ba57a7f94c3",
    ),
    "efficient_sam3_tinyvit_l": _stage1(
        "efficient_sam3_tinyvit_l",
        1778451881,
        "8209c620d728624b10136696bb3ce30b97e83e476b9ec9f41b0472ee5be7b0c1",
    ),
    "efficientsam3_efficientvit": _full(
        "efficientsam3_efficientvit",
        468394477,
        "086b04b2e7da7cc98aa4621b70c7291608aa9d187357b98d03bd4d6533ed5a17",
    ),
    "efficientsam3_repvit": _full(
        "efficientsam3_repvit",
        482292909,
        "cf9dc34f7e7a4fd6967e21ce660d45f7d6512669556c02658eadf469bdae20ab",
    ),
    "efficientsam3_tinyvit": _full(
        "efficientsam3_tinyvit",
        492809614,
        "3a52c42f975a9562cb656a57aebb60314d7871c3956e4955c91364a8fe3875dc",
    ),
    "sam3": HubFile(
        "facebook/sam3",
        "3c879f39826c281e95690f02c7821c4de09afae7",
        "sam3.pt",
        PinnedFile(
            3450062241,
            "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e",
        ),
    ),
}

COCO_ANNOTATIONS_ARCHIVE = PinnedFile(
    252907541, "113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268"
)
SACO_GOLD_ARCHIVE = PinnedFile(
    2716161466, "5ca9be2ea826e29a23f8dafd1520e5ffea2d2753fc901283874402ebf2bad6b4"
)
COCO_VAL_ANNOTATIONS = PinnedFile(
    19987840, "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f"
)


def sha256_of(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matches(path: Path, pin: PinnedFile) -> bool:
    """Return whether path is a file with the pinned size and SHA-256."""
    return (
        path.is_file()
        and path.stat().st_size == pin.size_bytes
        and sha256_of(path) == pin.sha256
    )


def require(path: Path, pin: PinnedFile) -> Path:
    """Return path if it matches pin; otherwise delete it and raise.

    Raises:
        ChecksumError: The file is missing or differs from the pin.
    """
    if not matches(path, pin):
        path.unlink(missing_ok=True)
        raise ChecksumError(f"{path} does not match its pinned size and SHA-256")
    return path


def stream_to_file(url: str, target: Path) -> Path:
    """Download url to target in fixed-size chunks, so memory use stays flat."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as out:
        for chunk in iter(lambda: response.read(CHUNK_BYTES), b""):
            out.write(chunk)
    return partial.replace(target)


def checkpoint_path(name: str, root: Path) -> Path:
    """Return where download_checkpoint stores a catalogued checkpoint.

    Raises:
        KeyError: name is not in CHECKPOINTS.
    """
    hub_file = CHECKPOINTS[name]
    return root / hub_file.repo_id.replace("/", "--") / hub_file.filename


def download_checkpoint(name: str, root: Path) -> Path:
    """Download a catalogued checkpoint, skipping it if a verified copy exists.

    Args:
        name: Key in CHECKPOINTS.
        root: Directory that holds all checkpoints.

    Returns:
        Path to the verified checkpoint.

    Raises:
        KeyError: name is not in CHECKPOINTS.
        ChecksumError: The downloaded file differs from its pin.
    """
    hub_file = CHECKPOINTS[name]
    target = checkpoint_path(name, root)
    local_dir = root / hub_file.repo_id.replace("/", "--")
    if matches(target, hub_file.pin):
        return target
    from huggingface_hub import constants, hf_hub_download

    # Plain HTTP streams to disk; Xet transfers can hold the whole file in memory.
    constants.HF_HUB_DISABLE_XET = True
    path = hf_hub_download(
        hub_file.repo_id,
        hub_file.filename,
        revision=hub_file.revision,
        local_dir=local_dir,
    )
    return require(Path(path), hub_file.pin)


def download_coco_annotations(coco_root: Path) -> Path:
    """Download and verify instances_val2017.json, skipping it if already verified.

    Returns:
        Path to the verified annotations file.

    Raises:
        ChecksumError: The archive or the extracted file differs from its pin.
    """
    target = coco_root / COCO_ANNOTATIONS_MEMBER
    if matches(target, COCO_VAL_ANNOTATIONS):
        return target
    archive = stream_to_file(
        COCO_ANNOTATIONS_URL, coco_root / "annotations_trainval2017.zip"
    )
    require(archive, COCO_ANNOTATIONS_ARCHIVE)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extract(COCO_ANNOTATIONS_MEMBER, coco_root)
    archive.unlink()
    return require(target, COCO_VAL_ANNOTATIONS)


def _images(annotations: Path) -> list[dict[str, Any]]:
    return json.loads(annotations.read_text())["images"]


def first_image_ids(annotations: Path, count: int) -> list[int]:
    """Return the first count image ids in file order.

    This is the order of COCO.getImgIds(), which eval_coco.py slices with
    --num_samples.
    """
    return [image["id"] for image in _images(annotations)[:count]]


def download_coco_images(
    annotations: Path, image_ids: list[int], coco_root: Path
) -> list[Path]:
    """Download the listed val2017 images, skipping files already present.

    Returns:
        Paths of the images, in image_ids order.
    """
    file_names = {image["id"]: image["file_name"] for image in _images(annotations)}
    return download_coco_files(
        [file_names[image_id] for image_id in image_ids], coco_root
    )


def download_coco_files(file_names: list[str], coco_root: Path) -> list[Path]:
    """Download the named val2017 image files, skipping files already present.

    Returns:
        Paths of the images, in file_names order.
    """
    image_dir = coco_root / "images" / "val2017"
    paths = []
    for file_name in file_names:
        target = image_dir / file_name
        if not target.is_file():
            stream_to_file(COCO_IMAGES_URL + file_name, target)
        paths.append(target)
    return paths


def download_saco_gold(root: Path) -> Path:
    """Download and unpack SA-Co/Gold, skipping it if already unpacked.

    The archive holds one zip per part, and each part unpacks to root/all/<part>,
    the layout upstream's SA-Co/Gold scripts read. The zips are deleted after.

    Returns:
        The directory that holds the parts, root/all.

    Raises:
        ChecksumError: The archive differs from its pin.
    """
    parts_root = root / "all"
    marker = parts_root / "complete.json"
    done = {"sha256": SACO_GOLD_ARCHIVE.sha256}
    if marker.is_file() and json.loads(marker.read_text()) == done:
        return parts_root
    archive = root / "all.zip"
    if not matches(archive, SACO_GOLD_ARCHIVE):
        require(stream_to_file(SACO_GOLD_URL, archive), SACO_GOLD_ARCHIVE)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(root)
    for part in SACO_GOLD_PARTS:
        inner = parts_root / f"{part}.zip"
        with zipfile.ZipFile(inner) as bundle:
            bundle.extractall(parts_root)
        inner.unlink()
    archive.unlink()
    marker.write_text(json.dumps(done))
    return parts_root
