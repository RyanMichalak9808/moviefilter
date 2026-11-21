"""Ask a Plex Media Server to rescan a folder after a file changed."""
import xml.etree.ElementTree as ET
from pathlib import PurePath

from .util import log


def _get(url, token, path, params=None):
    import requests
    r = requests.get(url.rstrip("/") + path, params=params or {},
                     headers={"X-Plex-Token": token, "Accept": "application/xml"}, timeout=20)
    r.raise_for_status()
    return r


def find_section(url, token, file_path):
    """Library section whose folder contains file_path (longest match), else None."""
    root = ET.fromstring(_get(url, token, "/library/sections").text)
    best, best_len = None, -1
    fp = str(PurePath(file_path))
    for d in root.iter("Directory"):
        for loc in d.iter("Location"):
            p = loc.get("path", "").rstrip("/\\")
            if p and fp.startswith(p) and len(p) > best_len:
                best, best_len = d.get("key"), len(p)
    return best


def refresh(url, token, file_path, section=None):
    """Partial scan of the file's folder (Plex re-analyzes changed files, picking up the new track)."""
    if not url or not token:
        return False
    try:
        section = section or find_section(url, token, file_path)
        if not section:
            log("  Plex: couldn't match the file to a library (paths differ in Docker?). "
                "Pass --plex-section ID.")
            return False
        folder = str(PurePath(file_path).parent)
        _get(url, token, f"/library/sections/{section}/refresh", {"path": folder})
        log(f"  Plex: rescan requested for section {section}: {folder}")
        return True
    except Exception as e:  # never fail the whole run because Plex is unreachable
        log(f"  Plex refresh failed: {e}")
        return False
