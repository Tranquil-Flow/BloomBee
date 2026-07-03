from __future__ import annotations


def test_torchdisk_does_not_spawn_cuda_copy_workers_when_cuda_unavailable(monkeypatch, tmp_path):
    from bloombee.flexgen_utils import pytorch_backend as backend

    monkeypatch.setattr(backend.torch.cuda, "is_available", lambda: False)
    created_threads = []

    class FakeThread:
        def __init__(self, *args, **kwargs):
            created_threads.append((args, kwargs))

        def start(self):  # pragma: no cover - should not be called
            raise AssertionError("CUDA copy worker thread should not start without CUDA")

    monkeypatch.setattr(backend.threading, "Thread", FakeThread)

    disk = backend.TorchDisk(tmp_path / "offload")

    assert disk.copy_threads == []
    assert created_threads == []
