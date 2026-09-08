import http.server
import threading

import pytest

from datacore.download import download_file, download_shards


@pytest.fixture
def http_server(tmp_path):
    (tmp_path / "shard_00.bin").write_bytes(b"hello-0")
    (tmp_path / "shard_01.bin").write_bytes(b"hello-1")

    handler = lambda *a, **kw: http.server.SimpleHTTPRequestHandler(*a, directory=str(tmp_path), **kw)
    server = http.server.HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    thread.join(timeout=5)


def test_download_file_writes_content(http_server, tmp_path):
    dest = tmp_path / "out" / "shard_00.bin"
    ok = download_file(f"{http_server}/shard_00.bin", str(dest))
    assert ok
    assert dest.read_bytes() == b"hello-0"


def test_download_file_skips_existing(http_server, tmp_path):
    dest = tmp_path / "out" / "shard_00.bin"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"already-here")
    ok = download_file(f"{http_server}/shard_00.bin", str(dest))
    assert ok
    assert dest.read_bytes() == b"already-here"  # not overwritten


def test_download_file_no_tmp_leftover_on_success(http_server, tmp_path):
    dest = tmp_path / "out" / "shard_00.bin"
    download_file(f"{http_server}/shard_00.bin", str(dest))
    assert not (dest.parent / (dest.name + ".tmp")).exists()


def test_download_file_returns_false_after_retries_exhausted(http_server, tmp_path, monkeypatch):
    monkeypatch.setattr("datacore.download.time.sleep", lambda s: None)  # skip real backoff delay
    ok = download_file(f"{http_server}/does-not-exist.bin", str(tmp_path / "x.bin"), max_attempts=2)
    assert ok is False
    assert not (tmp_path / "x.bin").exists()
    assert not (tmp_path / "x.bin.tmp").exists()


def test_download_shards(http_server, tmp_path):
    dest_dir = tmp_path / "shards"
    result = download_shards(
        http_server + "/shard_{index:02d}.bin", [0, 1], str(dest_dir),
        filename_fn=lambda i: f"shard_{i:02d}.bin",
    )
    assert result == {"successful": 2, "total": 2}
    assert (dest_dir / "shard_00.bin").read_bytes() == b"hello-0"
    assert (dest_dir / "shard_01.bin").read_bytes() == b"hello-1"
