from pathlib import Path

import pytest

from efficientsam3_jetson_benchmark import __main__ as cli
from efficientsam3_jetson_benchmark import download


def test_command_is_required() -> None:
    with pytest.raises(SystemExit):
        cli.main([])


def test_unknown_checkpoint_is_rejected() -> None:
    with pytest.raises(SystemExit):
        cli.main(["download", "checkpoint", "not-a-model"])


def test_download_checkpoint_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = []

    def fake_download(name: str, root: Path) -> Path:
        calls.append((name, root))
        return root / f"{name}.pt"

    monkeypatch.setattr(download, "download_checkpoint", fake_download)

    cli.main(["download", "checkpoint", "sam3", "--root", str(tmp_path)])

    assert calls == [("sam3", tmp_path)]
    assert capsys.readouterr().out.strip() == str(tmp_path / "sam3.pt")


def test_download_coco_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    annotations = tmp_path / "instances_val2017.json"
    monkeypatch.setattr(download, "download_coco_annotations", lambda root: annotations)
    monkeypatch.setattr(download, "first_image_ids", lambda path, count: [1, 2, 3])
    monkeypatch.setattr(
        download,
        "download_coco_images",
        lambda path, ids, root: [root / f"{image_id}.jpg" for image_id in ids],
    )

    cli.main(["download", "coco", "--first", "3", "--root", str(tmp_path)])

    assert capsys.readouterr().out.strip() == (
        f"3 images in {tmp_path / 'images' / 'val2017'}"
    )
