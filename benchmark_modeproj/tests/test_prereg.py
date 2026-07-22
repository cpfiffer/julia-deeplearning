"""Tests for the preregistration gate (numpy-free; stdlib only)."""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PREREG = os.path.join(os.path.dirname(HERE), "preregister.py")


def test_gate_refuses_without_split():
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "prereg.json")
        missing = os.path.join(d, "does_not_exist.json")
        r = subprocess.run([sys.executable, PREREG, "--split-file", missing, "--out", out],
                           capture_output=True, text=True)
        assert r.returncode != 0, "gate must fail when the split file is missing"
        assert not os.path.exists(out), "gate must not emit prereg.json on failure"
        assert "PREREG GATE FAILED" in r.stderr


def test_gate_emits_and_records_hash():
    with tempfile.TemporaryDirectory() as d:
        split = os.path.join(d, "frozen_split.json")
        with open(split, "w") as f:
            json.dump({"train": [1, 2], "val": [3], "test": [4, 5]}, f)
        out = os.path.join(d, "prereg.json")
        r = subprocess.run([sys.executable, PREREG, "--split-file", split, "--out", out],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        rec = json.load(open(out))
        assert rec["frozen_split"]["sha256"], "split hash must be recorded"
        assert rec["win_rule"]["statement"].startswith("CANDIDATE wins iff")
        assert rec["primary_decision_statistic"]["seeds"] == [0, 1, 2, 3, 4]
        # loss lambdas must start unfrozen so a downgrade cannot pass silently
        assert rec["loss_lambdas"]["frozen"] is False


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok:", fn.__name__)
    print(f"\nall {len(fns)} prereg tests passed")
