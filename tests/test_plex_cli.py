import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from moviefilter import cli, plex
from moviefilter.util import fmt_time, merge_ranges, parse_range, parse_time

SECTIONS = """<?xml version="1.0"?><MediaContainer>
<Directory key="1" title="Movies"><Location path="/data/movies"/></Directory>
<Directory key="2" title="Kids"><Location path="/data/movies/kids"/></Directory>
</MediaContainer>"""


@pytest.fixture
def server():
    seen = []

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append((self.path, self.headers.get("X-Plex-Token")))
            body = SECTIONS.encode() if self.path.startswith("/library/sections?") or self.path == "/library/sections" else b""
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}", seen
    srv.shutdown()


def test_plex_refresh_picks_most_specific_section(server, monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    url, seen = server
    assert plex.refresh(url, "tok", "/data/movies/kids/Up (2009)/Up (2009).mkv")
    assert seen[-1][0].startswith("/library/sections/2/refresh?path=%2Fdata%2Fmovies%2Fkids%2FUp+%282009%29")
    assert all(tok == "tok" for _, tok in seen)


def test_plex_failure_never_raises():
    assert plex.refresh("http://127.0.0.1:1", "tok", "/x/y.mkv", section="1") is False
    assert plex.refresh(None, None, "/x.mkv") is False


def test_cli_parses_all_commands():
    p = cli.build_parser()
    a = p.parse_args(["analyze", "m.mkv", "--levels", "all", "--stem-transcribe", "full", "--also", "twerp"])
    assert a.levels == ["strong", "slurs", "mild", "religious"] and a.also == ["twerp"]
    a = p.parse_args(["apply", "m.mkv", "--in-place", "--make-default", "--codec", "eac3"])
    assert a.in_place and a.cuts is None
    a = p.parse_args(["review", "m.mkv", "--drop", "3", "5", "--add", "1:02:03.5-1:02:04"])
    assert a.drop == [3, 5]
    a = p.parse_args(["batch", "/movies", "--in-place", "--dry-run"])
    assert a.dry_run
    with pytest.raises(SystemExit):
        p.parse_args(["apply", "m.mkv"])                  # needs -o or --in-place
    with pytest.raises(SystemExit):
        p.parse_args(["analyze", "m.mkv", "--levels", "rude"])


def test_time_helpers():
    assert parse_time("1:02:03.5") == pytest.approx(3723.5)
    assert parse_time("83") == 83
    assert parse_range("1:00-1:02.5") == (60, 62.5)
    with pytest.raises(ValueError):
        parse_range("1:05-1:00")
    assert fmt_time(3723.5) == "1:02:03.50"
    m = merge_ranges([{"start": 0, "end": 1, "level": "strong"}, {"start": 1.1, "end": 2, "level": "mild"}], gap=0.2)
    assert len(m) == 1 and m[0]["levels"] == ["strong", "mild"]
