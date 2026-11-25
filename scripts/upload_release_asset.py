#!/usr/bin/env python3
"""Upload a file as an asset to a GitHub Release using the repo's configured PAT.
Never prints the token. Usage:
    python3 scripts/upload_release_asset.py <tag> <file_path> [release_name]
"""
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request


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


def gh_req(method: str, url: str, token: str, data=None, content_type="application/json"):
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "blockfish-colab-worker",
    }
    if data is not None and content_type:
        headers["Content-Type"] = content_type
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=300) as resp:
        raw = resp.read()
        return resp.status, json.loads(raw.decode("utf-8")) if raw else {}


def main():
    if len(sys.argv) < 3:
        print("Usage: upload_release_asset.py <tag> <file_path> [release_name]", file=sys.stderr)
        sys.exit(1)

    tag = sys.argv[1]
    file_path = sys.argv[2]
    release_name = sys.argv[3] if len(sys.argv) > 3 else f"Blockfish Dataset ({tag})"
    repo = "jush0147/blockfish"

    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    token = get_token()
    asset_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    print(f"Uploading {asset_name} ({file_size / (1024 * 1024):.2f} MB) to release tag '{tag}'...")

    # 1. Get or create release by tag
    rel_url = f"https://api.github.com/repos/{repo}/releases/tags/{urllib.parse.quote(tag)}"
    try:
        _, rel = gh_req("GET", rel_url, token)
        print(f"Found existing release id={rel['id']} ({rel['html_url']})")
    except urllib.error.HTTPError as e:
        if e.code != 404:
            print(f"GitHub API error getting release: HTTP {e.code}", file=sys.stderr)
            sys.exit(1)
        payload = json.dumps({
            "tag_name": tag,
            "target_commitish": "rescue/independent-validation-20261116",
            "name": release_name,
            "body": "Auto-uploaded dataset/weights from Blockfish Colab Worker.",
            "draft": False,
            "prerelease": True,
        }).encode("utf-8")
        _, rel = gh_req("POST", f"https://api.github.com/repos/{repo}/releases", token, data=payload)
        print(f"Created release id={rel['id']} ({rel['html_url']})")

    # 2. Delete existing asset with same name if present
    for asset in rel.get("assets", []):
        if asset.get("name") == asset_name:
            del_url = f"https://api.github.com/repos/{repo}/releases/assets/{asset['id']}"
            gh_req("DELETE", del_url, token)
            print(f"Replaced existing asset '{asset_name}' (id={asset['id']})")

    # 3. Upload asset
    upload_base = rel["upload_url"].split("{")[0]
    upload_url = f"{upload_base}?name={urllib.parse.quote(asset_name)}"
    with open(file_path, "rb") as f:
        blob = f.read()
    _, uploaded = gh_req(
        "POST",
        upload_url,
        token,
        data=blob,
        content_type="application/gzip" if asset_name.endswith(".gz") else "application/octet-stream",
    )
    print(f"UPLOAD SUCCESS: {uploaded.get('browser_download_url')} ({uploaded.get('size')} bytes)")


if __name__ == "__main__":
    main()
