import hashlib
import json
from pathlib import Path

import pytest

from efficientsam3_jetson_benchmark.download import PinnedFile


@pytest.fixture
def pinned_file(tmp_path: Path) -> tuple[Path, PinnedFile]:
    """A small file and the pin that matches it."""
    content = b"checkpoint bytes"
    path = tmp_path / "file.bin"
    path.write_bytes(content)
    return path, PinnedFile(len(content), hashlib.sha256(content).hexdigest())


@pytest.fixture
def annotations_file(tmp_path: Path) -> Path:
    """A minimal COCO annotations file whose image order differs from id order."""
    dataset = {
        "images": [
            {"id": 30, "file_name": "000000000030.jpg"},
            {"id": 10, "file_name": "000000000010.jpg"},
            {"id": 20, "file_name": "000000000020.jpg"},
        ],
        "annotations": [],
        "categories": [],
    }
    path = tmp_path / "instances_val2017.json"
    path.write_text(json.dumps(dataset))
    return path
