"""Recompute ECR's AUC at any evaluation batch size, from the Patch 3 dump.

ECR's evaluate_rec.py:55-72 computes AUC inside each batch (every liked item against every
disliked item, ties counting half), averages those per-batch values, and silently drops any
batch that has no liked or no disliked item. This replays that from the logged scores, so
one GPU run gives the whole curve.

The valid/test dataloaders are built without shuffle (train_rec.py:225-234), so example
order is the file order and re-grouping offline reproduces what a real run at that batch
size would see.
"""
import json
import os
import sys

DUMP = sys.argv[1] if len(sys.argv) > 1 else "save/redial_rec/auc_dump.jsonl"
PHASE = {0: "valid", 1: "test"}


def auc_brute(pos, neg):
    """ECR's own double loop. O(len(pos)*len(neg)) -- only for small batches."""
    a = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                a += 1
            elif p == n:
                a += 0.5
    return a / (len(pos) * len(neg))


def auc_fast(pos, neg):
    """Same value via mid-ranks (Mann-Whitney U). Handles ties identically."""
    allv = sorted(pos + neg)
    # mid-rank of each distinct value
    rank = {}
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        mid = (i + j) / 2.0 + 1.0
        rank[allv[i]] = mid
        i = j + 1
    r = sum(rank[p] for p in pos)
    npos, nneg = len(pos), len(neg)
    return (r - npos * (npos + 1) / 2.0) / (npos * nneg)


def sweep(rows, bs):
    """ECR's batch-averaged AUC at batch size bs. Returns (auc, kept, dropped, used_ex)."""
    total, kept, dropped, used = 0.0, 0, 0, 0
    for start in range(0, len(rows), bs):
        chunk = rows[start:start + bs]
        pos = [r["score"] for r in chunk if r["sent"] == 1]
        neg = [r["score"] for r in chunk if r["sent"] == 0]
        if not pos or not neg:          # ECR's bare `except: pass`
            dropped += 1
            continue
        total += auc_fast(pos, neg)
        kept += 1
        used += len(pos) + len(neg)
    return (total / kept if kept else float("nan")), kept, dropped, used


def main():
    if not os.path.exists(DUMP):
        sys.exit(f"missing {DUMP} -- run the evaluation with Patch 3 first")

    by_phase = {}
    for line in open(DUMP, encoding="utf-8"):
        r = json.loads(line)
        by_phase.setdefault(r["phase"], []).append(r)

    # self-check: the fast AUC must agree with ECR's literal double loop
    probe = by_phase[max(by_phase)][:400]
    p = [r["score"] for r in probe if r["sent"] == 1]
    n = [r["score"] for r in probe if r["sent"] == 0]
    if p and n:
        d = abs(auc_fast(p, n) - auc_brute(p, n))
        print(f"self-check vs ECR's double loop: difference {d:.2e}")
        assert d < 1e-9
    print()

    for ph in sorted(by_phase):
        rows = by_phase[ph]
        name = PHASE.get(ph, f"phase {ph}")
        npos = sum(1 for r in rows if r["sent"] == 1)
        nneg = sum(1 for r in rows if r["sent"] == 0)
        nnot = sum(1 for r in rows if r["sent"] not in (0, 1))
        print(f"=== {name}: {len(rows)} examples "
              f"({npos} liked, {nneg} disliked, {nnot} not stated)")
        print()
        print(f"{'batch':>7} {'AUC':>9} {'batches':>9} {'dropped':>9} "
              f"{'% ex. used':>11}")
        for bs in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, len(rows)]:
            a, kept, dropped, used = sweep(rows, bs)
            label = "all" if bs == len(rows) else str(bs)
            pct = 100.0 * used / len(rows)
            print(f"{label:>7} {a:>9.4f} {kept:>9} {dropped:>9} {pct:>10.1f}%")
        print()


if __name__ == "__main__":
    main()
