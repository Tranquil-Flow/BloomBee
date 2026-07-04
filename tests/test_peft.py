import os
import shutil

import pytest
from huggingface_hub import snapshot_download
from peft.utils import SAFETENSORS_WEIGHTS_NAME

import bloombee.utils.peft as peft_utils

UNSAFE_PEFT_REPO = "artek0chumak/bloom-560m-unsafe-peft"
SAFE_PEFT_REPO = "artek0chumak/bloom-560m-safe-peft"
TMP_CACHE_DIR = "tmp_cache/"

live_hf_peft = pytest.mark.skipif(
    os.environ.get("BLOOMBEE_RUN_HF_PEFT") != "1",
    reason="live HuggingFace PEFT tests require network/cache; set BLOOMBEE_RUN_HF_PEFT=1 to opt in",
)


def clear_dir(path_to_dir):
    shutil.rmtree(path_to_dir)
    os.mkdir(path_to_dir)


def dir_empty(path_to_dir):
    files = os.listdir(path_to_dir)
    return len(files) == 0


def test_check_peft_repository_checks_safetensors_weight_path(monkeypatch):
    seen_paths = []

    class FakeHfFileSystem:
        def exists(self, path):
            seen_paths.append(path)
            return path.endswith(SAFETENSORS_WEIGHTS_NAME)

    monkeypatch.setattr(peft_utils, "HfFileSystem", FakeHfFileSystem)

    assert peft_utils.check_peft_repository("org/repo") is True
    assert seen_paths == [f"org/repo/{SAFETENSORS_WEIGHTS_NAME}"]


def test_load_peft_rejects_unsafe_repo_before_cache_access(monkeypatch, tmp_path):
    monkeypatch.setattr(peft_utils, "check_peft_repository", lambda repo_id: False)

    def fail_if_called(*args, **kwargs):  # pragma: no cover - assertion helper
        raise AssertionError("unsafe PEFT repo should be rejected before adapter/cache access")

    monkeypatch.setattr(peft_utils, "get_adapter_from_repo", fail_if_called)

    with pytest.raises(ValueError, match="doesn't have safetensors"):
        peft_utils.load_peft(UNSAFE_PEFT_REPO, cache_dir=str(tmp_path), delay=0)


@live_hf_peft
def test_check_peft():
    assert not peft_utils.check_peft_repository(UNSAFE_PEFT_REPO), "NOSAFE_PEFT_REPO is safe to load."
    assert peft_utils.check_peft_repository(SAFE_PEFT_REPO), "SAFE_PEFT_REPO is not safe to load."


@live_hf_peft
def test_load_noncached(tmpdir):
    clear_dir(tmpdir)
    with pytest.raises(Exception):
        peft_utils.load_peft(UNSAFE_PEFT_REPO, cache_dir=tmpdir)

    assert dir_empty(tmpdir), "UNSAFE_PEFT_REPO is loaded"

    peft_utils.load_peft(SAFE_PEFT_REPO, cache_dir=tmpdir)

    assert not dir_empty(tmpdir), "SAFE_PEFT_REPO is not loaded"


@live_hf_peft
def test_load_cached(tmpdir):
    clear_dir(tmpdir)
    snapshot_download(SAFE_PEFT_REPO, cache_dir=tmpdir)

    peft_utils.load_peft(SAFE_PEFT_REPO, cache_dir=tmpdir)


@live_hf_peft
def test_load_layer_exists(tmpdir):
    clear_dir(tmpdir)

    peft_utils.load_peft(SAFE_PEFT_REPO, block_idx=2, cache_dir=tmpdir)


@live_hf_peft
def test_load_layer_nonexists(tmpdir):
    clear_dir(tmpdir)

    peft_utils.load_peft(
        SAFE_PEFT_REPO,
        block_idx=1337,
        cache_dir=tmpdir,
    )
