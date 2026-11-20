"""Measure cross-game coverage of two-placement digs. Run from repo root."""
import hashlib
import json
from pathlib import Path

RESULTS = Path("research/results")
REFLECT = str.maketrans("JLSZ", "LJZS")


def surface(observation):
    cells = observation["cells"]
    floor = max((r for (r, _), color in cells if color == "G"), default=None)
    if floor is None:
        return None
    rows = [0] * (max(r for (r, _), _ in cells) - floor + 1)
    for (r, c), _ in cells:
        if r >= floor:
            rows[r - floor] |= 1 << c
    return tuple(rows)


def keys(observation):
    board = surface(observation)
    if board is None:
        return None
    mirrored = tuple(int(format(row, "010b")[::-1], 2) for row in board)
    pieces = observation["current"] + (observation["held"] or "-") + observation["next"][0]
    native = board, pieces
    reflected = mirrored, pieces.translate(REFLECT)
    return min(native, reflected), min(board, mirrored), reflected < native


def main():
    audit = json.loads((RESULTS / "s34-replay-audit.json").read_text())
    records = sorted(audit["records"], key=lambda record: record["seed"])
    assert [record["seed"] for record in records] == list(range(76800, 76832))
    library = {}
    geometry = set()
    examples = []
    queries = []
    for record in records:
        raw = Path(record["trace"]).read_bytes()
        assert record["returncode"] == 0 and hashlib.sha256(raw).hexdigest() == record["sha256"]
        game = json.loads(raw)
        events = game["world_events"]
        clears = game["ds"]
        assert len(events) == len(clears) == len(game["inputs"]) and clears[-1] == 100
        assert [event["cleared_before"] for event in events] == [0] + clears[:-1]
        seed = record["seed"]
        for step, event in enumerate(events):
            index = keys(event["observation"])
            if seed < 76816:
                if index is None or step + 1 >= len(events):
                    continue
                gain = clears[step + 1] - event["cleared_before"]
                if gain < 2:
                    continue
                key, board, mirrored = index
                library.setdefault(key, []).append(len(examples))
                geometry.add(board)
                examples.append(dict(seed=seed, step=step, mirrored=mirrored, gain=gain,
                                     surface=key[0], pieces=key[1], actions=game["inputs"][step:step+2]))
            else:
                hits = library.get(index[0], []) if index else []
                queries.append(dict(seed=seed, step=step, examples=hits,
                                    geometry=bool(index and index[1] in geometry)))
    matched = sum(bool(q["examples"]) for q in queries)
    summary = dict(examples=len(examples), keys=len(library), queries=len(queries),
                   matched=matched, geometry=sum(q["geometry"] for q in queries),
                   games_matched=len({q["seed"] for q in queries if q["examples"]}),
                   passed=matched >= 16)
    (RESULTS / "s35-probe.json").write_text(json.dumps(dict(summary=summary, examples=examples,
                                                           queries=queries), separators=(",", ":")) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
