#!/usr/bin/env python3
"""Download a named asset from a GitHub Release using the repo's configured PAT.
Strips Authorization header on cross-host redirects (required for GitHub S3/Azure asset URLs).
Never prints the token. Usage:
    python3 scripts/download_release_asset.py <tag> <asset_name> <dest_path>
"""
import json
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request


class SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_req is not None:
            old_host = urllib.parse.urlparse(req.full_url).netloc
            new_host = urllib.parse.urlparse(newurl).netloc
            if old_host != new_host:
                new_req.remove_header("Authorization")
        return new_req


def get_token() -> str:
    tok = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if tok:
        return tok.strip()
    url = subprocess.check_output(
        ["git", "remote", "get-url", "origin"], text=True
    ).strip()
    m = re.match(r"https://(?:[^:@]+:)?([^@]+)@github\.com/", url)
    if not m:
        raise RuntimeError("Could not extract GitHub token from git remote origin URL")
    return m.group(1)


def main():
    if len(sys.argv) < 4:
        print("Usage: download_release_asset.py <tag> <asset_name> <dest_path>", file=sys.stderr)
        sys.exit(1)

    tag, asset_name, dest_path = sys.argv[1], sys.argv[2], sys.argv[3]
    repo = "jush0147/blockfish"
    token = get_token()
    opener = urllib.request.build_opener(SafeRedirectHandler())

    rel_url = f"https://api.github.com/repos/{repo}/releases/tags/{urllib.parse.quote(tag)}"
    req = urllib.request.Request(
        rel_url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "blockfish-colab-worker",
        },
    )
    with opener.open(req, timeout=60) as resp:
        rel = json.loads(resp.read().decode("utf-8"))

    asset_id = None
    for a in rel.get("assets", []):
        if a.get("name") == asset_name:
            asset_id = a["id"]
            break
    if asset_id is None:
        print(f"Asset '{asset_name}' not found in release '{tag}'", file=sys.stderr)
        sys.exit(1)

    dl_url = f"https://api.github.com/repos/{repo}/releases/assets/{asset_id}"
    dl_req = urllib.request.Request(
        dl_url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/octet-stream",
            "User-Agent": "blockfish-colab-worker",
        },
    )
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    print(f"Downloading '{asset_name}' from release '{tag}' -> {dest_path}...", flush=True)
    with opener.open(dl_req, timeout=300) as resp, open(dest_path, "wb") as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    sz = os.path.getsize(dest_path)
    print(f"DOWNLOAD SUCCESS: {dest_path} ({sz / (1024 * 1024):.2f} MB)", flush=True)


if __name__ == "__main__":
    main()
