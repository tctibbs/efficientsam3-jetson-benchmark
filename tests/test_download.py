import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import pytest
from huggingface_hub import constants
from pycocotools.coco import COCO

from efficientsam3_jetson_benchmark import download
from efficientsam3_jetson_benchmark.download import ChecksumError, HubFile, PinnedFile


def _pin(content: bytes) -> PinnedFile:
    return PinnedFile(len(content), hashlib.sha256(content).hexdigest())


@pytest.fixture
def fake_checkpoint(monkeypatch: pytest.MonkeyPatch) -> tuple[str, bytes]:
    """Register a small checkpoint in the catalog and return its name and bytes."""
    content = b"weights"
    hub_file = HubFile("owner/repo", "abc123", "nested/model.pt", _pin(content))
    monkeypatch.setitem(download.CHECKPOINTS, "fake", hub_file)
    return "fake", content


def test_sha256_of_matches_hashlib(pinned_file: tuple[Path, PinnedFile]) -> None:
    path, pin = pinned_file

    assert download.sha256_of(path) == pin.sha256


@pytest.mark.parametrize(
    ("size_delta", "sha256", "expected"),
    [
        (0, None, True),
        (1, None, False),
        (0, "0" * 64, False),
    ],
)
def test_matches_checks_size_and_hash(
    pinned_file: tuple[Path, PinnedFile],
    size_delta: int,
    sha256: str | None,
    expected: bool,
) -> None:
    path, pin = pinned_file
    candidate = PinnedFile(pin.size_bytes + size_delta, sha256 or pin.sha256)

    assert download.matches(path, candidate) is expected


def test_matches_is_false_for_missing_file(tmp_path: Path) -> None:
    assert download.matches(tmp_path / "missing", PinnedFile(1, "0" * 64)) is False


def test_require_returns_matching_path(pinned_file: tuple[Path, PinnedFile]) -> None:
    path, pin = pinned_file

    assert download.require(path, pin) == path


def test_require_deletes_mismatched_file(pinned_file: tuple[Path, PinnedFile]) -> None:
    path, pin = pinned_file

    with pytest.raises(ChecksumError, match="does not match"):
        download.require(path, PinnedFile(pin.size_bytes, "0" * 64))
    assert not path.exists()


def test_stream_to_file_writes_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        download.urllib.request, "urlopen", lambda url, timeout: io.BytesIO(b"data")
    )
    target = tmp_path / "sub" / "file.jpg"

    result = download.stream_to_file("http://example.invalid/file.jpg", target)

    assert result == target
    assert target.read_bytes() == b"data"
    assert not target.with_name("file.jpg.part").exists()


def test_download_checkpoint_skips_verified_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_checkpoint: tuple[str, bytes],
) -> None:
    name, content = fake_checkpoint
    target = tmp_path / "owner--repo" / "nested" / "model.pt"
    target.parent.mkdir(parents=True)
    target.write_bytes(content)

    def fail(*args: object, **kwargs: object) -> str:
        raise AssertionError("downloaded a file that was already verified")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fail)

    assert download.download_checkpoint(name, tmp_path) == target


def test_download_checkpoint_streams_without_xet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_checkpoint: tuple[str, bytes],
) -> None:
    name, content = fake_checkpoint
    monkeypatch.setattr(constants, "HF_HUB_DISABLE_XET", False)
    calls = []

    def fake_hf_hub_download(
        repo_id: str, filename: str, revision: str, local_dir: Path
    ) -> str:
        calls.append((repo_id, filename, revision, constants.HF_HUB_DISABLE_XET))
        path = local_dir / filename
        path.parent.mkdir(parents=True)
        path.write_bytes(content)
        return str(path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)

    result = download.download_checkpoint(name, tmp_path)

    assert result == tmp_path / "owner--repo" / "nested" / "model.pt"
    assert calls == [("owner/repo", "nested/model.pt", "abc123", True)]


def test_download_checkpoint_rejects_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_checkpoint: tuple[str, bytes],
) -> None:
    name, _ = fake_checkpoint

    def corrupt_download(
        repo_id: str, filename: str, revision: str, local_dir: Path
    ) -> str:
        path = local_dir / filename
        path.parent.mkdir(parents=True)
        path.write_bytes(b"corrupt")
        return str(path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", corrupt_download)

    with pytest.raises(ChecksumError):
        download.download_checkpoint(name, tmp_path)
    assert not (tmp_path / "owner--repo" / "nested" / "model.pt").exists()


@pytest.mark.parametrize(("name", "hub_file"), list(download.CHECKPOINTS.items()))
def test_catalog_entries_are_pinned(name: str, hub_file: HubFile) -> None:
    assert re.fullmatch(r"[0-9a-f]{64}", hub_file.pin.sha256)
    assert re.fullmatch(r"[0-9a-f]{40}", hub_file.revision)
    assert hub_file.pin.size_bytes > 0
    assert Path(hub_file.filename).stem == name


@pytest.mark.parametrize(
    "name", [name for name in download.CHECKPOINTS if name.startswith("efficient_sam3")]
)
def test_stage1_names_parse_like_eval_coco(name: str) -> None:
    # eval_coco.py splits the file name on "_" and reads the backbone and size.
    parts = name.split("_")

    assert parts[2] in {"efficientvit", "repvit", "tinyvit"}
    assert parts[3] in {"s", "m", "l"}


def test_first_image_ids_match_pycocotools(annotations_file: Path) -> None:
    expected = COCO(str(annotations_file)).getImgIds()[:2]

    assert download.first_image_ids(annotations_file, 2) == expected == [30, 10]


def test_download_coco_images_fetches_only_missing(
    tmp_path: Path, annotations_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_dir = tmp_path / "images" / "val2017"
    image_dir.mkdir(parents=True)
    (image_dir / "000000000030.jpg").write_bytes(b"present")
    fetched = []

    def fake_stream(url: str, target: Path) -> Path:
        fetched.append(url)
        target.write_bytes(b"fetched")
        return target

    monkeypatch.setattr(download, "stream_to_file", fake_stream)

    paths = download.download_coco_images(annotations_file, [30, 10], tmp_path)

    assert paths == [image_dir / "000000000030.jpg", image_dir / "000000000010.jpg"]
    assert fetched == [download.COCO_IMAGES_URL + "000000000010.jpg"]


def _annotations_archive() -> tuple[bytes, bytes]:
    member = b'{"images": []}'
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr(download.COCO_ANNOTATIONS_MEMBER, member)
    return buffer.getvalue(), member


def test_download_coco_annotations_extracts_and_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, member = _annotations_archive()
    monkeypatch.setattr(download, "COCO_ANNOTATIONS_ARCHIVE", _pin(archive))
    monkeypatch.setattr(download, "COCO_VAL_ANNOTATIONS", _pin(member))

    def fake_stream(url: str, target: Path) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive)
        return target

    monkeypatch.setattr(download, "stream_to_file", fake_stream)

    result = download.download_coco_annotations(tmp_path)

    assert result.read_bytes() == member
    assert not (tmp_path / "annotations_trainval2017.zip").exists()


def test_download_coco_annotations_skips_verified_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / download.COCO_ANNOTATIONS_MEMBER
    target.parent.mkdir(parents=True)
    target.write_bytes(b"{}")
    monkeypatch.setattr(download, "COCO_VAL_ANNOTATIONS", _pin(b"{}"))

    def fail(url: str, target: Path) -> Path:
        raise AssertionError("downloaded annotations that were already verified")

    monkeypatch.setattr(download, "stream_to_file", fail)

    assert download.download_coco_annotations(tmp_path) == target


def _saco_gold_archive() -> bytes:
    """An all.zip that nests one zip per part, as the SA-Co/Gold archive does."""
    outer = io.BytesIO()
    with zipfile.ZipFile(outer, "w") as bundle:
        for part in download.SACO_GOLD_PARTS:
            inner = io.BytesIO()
            with zipfile.ZipFile(inner, "w") as part_bundle:
                part_bundle.writestr(f"{part}/{part}.txt", part)
            bundle.writestr(f"all/{part}.zip", inner.getvalue())
    return outer.getvalue()


def test_download_saco_gold_unpacks_each_part(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = _saco_gold_archive()
    monkeypatch.setattr(download, "SACO_GOLD_ARCHIVE", _pin(archive))

    def fake_stream(url: str, target: Path) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive)
        return target

    monkeypatch.setattr(download, "stream_to_file", fake_stream)

    parts_root = download.download_saco_gold(tmp_path)

    assert parts_root == tmp_path / "all"
    for part in download.SACO_GOLD_PARTS:
        assert (parts_root / part / f"{part}.txt").read_text() == part
    assert list(tmp_path.rglob("*.zip")) == []


def test_download_saco_gold_reuses_a_verified_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = _saco_gold_archive()
    monkeypatch.setattr(download, "SACO_GOLD_ARCHIVE", _pin(archive))
    (tmp_path / "all.zip").write_bytes(archive)

    def fail(url: str, target: Path) -> Path:
        raise AssertionError("downloaded an archive that was already verified")

    monkeypatch.setattr(download, "stream_to_file", fail)

    parts_root = download.download_saco_gold(tmp_path)

    assert (parts_root / "sa1b-images" / "sa1b-images.txt").is_file()


def test_download_saco_gold_skips_an_unpacked_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "all" / "complete.json"
    marker.parent.mkdir()
    marker.write_text(json.dumps({"sha256": download.SACO_GOLD_ARCHIVE.sha256}))

    def fail(url: str, target: Path) -> Path:
        raise AssertionError("downloaded SA-Co/Gold that was already unpacked")

    monkeypatch.setattr(download, "stream_to_file", fail)

    assert download.download_saco_gold(tmp_path) == tmp_path / "all"
