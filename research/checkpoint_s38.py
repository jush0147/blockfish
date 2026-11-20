"""Stage complete ledger records and their traces while the S38 runner is active."""
import json
import subprocess
from pathlib import Path

root = Path("research/results")
for ledger in sorted(root.glob("s38-*.jsonl")):
    raw = ledger.read_bytes()
    # A record is flushed after its trace and usage files are closed.
    prefix = raw[:raw.rfind(b"\n")+1]
    if not prefix:
        continue
    files = []
    for line in prefix.splitlines():
        row = json.loads(line)
        if row.get("trace"):
            trace = Path(row["trace"])
            for path in (trace,trace.with_suffix(".usage.json")):
                if path.is_file():
                    files.append(path)
    if files:
        subprocess.run(["git","add","--",*files],check=True)
    blob = subprocess.check_output(["git","hash-object","-w","--stdin"],input=prefix).decode().strip()
    subprocess.run(["git","update-index","--add","--cacheinfo",f"100644,{blob},{ledger}"],check=True)
    print(ledger,len(prefix.splitlines()))
