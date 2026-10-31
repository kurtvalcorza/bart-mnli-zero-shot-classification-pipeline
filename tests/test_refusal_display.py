"""DAT19/DAT20: a refused BYOD input shows one clean line, ``Refused: <file>: <reason>``, with no traceback.

The router cell carries ml-worker's refusal channel (template ``dat20-router-refusal/1``) and Section 4 wraps its
whole BYOD branch so that every user-input problem raises ``BYODRefusal``. These tests drive both from the
notebook JSON with CI's dependencies only:

- Section 4 is executed with a fake upload and a whitespace "tokenizer" for each refusal path;
- the router's kernel loop is driven with a queued stub connection on every platform;
- on POSIX the real worker process runs the notebook's own cells, so the full display path is checked end to end.
"""
# ruff: noqa: E501  -- notebook source literals are matched verbatim

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "bart_zero_shot_classification_colab.ipynb"
STEM = "bart_zero_shot_classification"

NB = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
CELLS = ["".join(c["source"]) if isinstance(c["source"], list) else c["source"] for c in NB["cells"]]
CODE = [i for i, c in enumerate(NB["cells"]) if c["cell_type"] == "code"]
MODULE_CELLS = [i for i in CODE if NB["cells"][i]["metadata"].get("dimer", {}).get("embedded_module")]


def _cell(marker: str) -> int:
    found = [i for i in CODE if marker in CELLS[i]]
    assert len(found) == 1, (marker, found)
    return found[0]


SEC4 = _cell("USE_BYOD = False  # @param")
ROUTER = _cell("_WORKER_SOURCE = r")

PIPE_STANDIN = (
    "class _Pipe:\n"
    "    adapter = None\n\n"
    "    def count_pair_tokens(self, text, labels, *, hypothesis_template):\n"
    "        return {label: len(text.split()) + len(hypothesis_template.format(label).split()) + 4 for label in labels}\n\n\n"
    "pipe = _Pipe()\n"
)


def _section4_source(fields: dict) -> str:
    src = CELLS[SEC4]
    for name, value in fields.items():
        src, n = re.subn(rf"^{name} = .*?(  # @param.*)$", lambda m, v=value, k=name: f"{k} = {v!r}{m.group(1)}", src, flags=re.M)
        assert n == 1, name
    return src


def _records(per_label: dict[str, int], words: int = 0) -> list[dict]:
    out, k = [], 0
    for label, n in per_label.items():
        for j in range(n):
            text = f"m{k} " + " ".join(["w"] * words) if words and j == 0 else f"message {k} about {label} number {j}"
            out.append({"id": f"r{k:05d}", "text": text, "label": label})
            k += 1
    return out


def _csv(records: list[dict]) -> bytes:
    rows = ["id,text,label"] + [f"{r['id']},\"{r['text']}\",{r['label']}" for r in records]
    return ("\n".join(rows) + "\n").encode("utf-8")


class _FakeFiles:
    payload: dict = {}

    @classmethod
    def upload(cls):
        return dict(cls.payload)


@pytest.fixture
def section4(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ns: dict = {"__name__": "__main__", "os": os}
    for index in MODULE_CELLS:
        exec(compile(CELLS[index], f"<cell {index}>", "exec"), ns)
    exec(PIPE_STANDIN, ns)

    def run(fields: dict, colab: bool = True):
        google = types.ModuleType("google")
        google.__path__ = []
        colab_mod = types.ModuleType("google.colab")
        colab_mod.files = _FakeFiles
        google.colab = colab_mod
        saved = {k: sys.modules.get(k) for k in ("google", "google.colab")}
        sys.modules["google"] = google
        if colab:
            sys.modules["google.colab"] = colab_mod
        else:
            sys.modules.pop("google.colab", None)
        out, old = io.StringIO(), sys.stdout
        sys.stdout = out
        try:
            exec(compile(_section4_source(fields), f"<cell {SEC4}>", "exec"), ns)
        finally:
            sys.stdout = old
            for key, value in saved.items():
                if value is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = value
        return ns, out.getvalue()

    run.ns = ns
    return run


def _refusal(run, fields: dict, colab: bool = True):
    with pytest.raises(ValueError) as info:
        run(fields, colab)
    exc = info.value
    assert getattr(exc, "dimer_refusal", False) is True, repr(exc)
    lines = exc._render_traceback_()
    assert len(lines) == 1 and lines[0].startswith("Refused: ") and "Traceback" not in lines[0], lines
    return lines[0]


# ---- each refusal path in Section 4 is one clean line naming the file ---------------------------------------


def test_byod_path_that_is_not_a_file_is_refused_by_path(section4, tmp_path):
    line = _refusal(section4, {"USE_BYOD": True, "BYOD_PATH": str(tmp_path / "missing.csv")})
    assert line.startswith(f"Refused: {tmp_path / 'missing.csv'}: BYOD_PATH is not a file")
    line = _refusal(section4, {"USE_BYOD": True, "BYOD_PATH": str(tmp_path)})
    assert "BYOD_PATH is not a file" in line


def test_cancelled_multi_file_and_non_colab_uploads_are_refused(section4):
    _FakeFiles.payload = {}
    assert _refusal(section4, {"USE_BYOD": True}).startswith("Refused: (no file): upload exactly one .csv, .json or .jsonl file (received 0)")
    _FakeFiles.payload = {"a.csv": b"x", "b.csv": b"y"}
    assert _refusal(section4, {"USE_BYOD": True}).startswith("Refused: a.csv, b.csv: upload exactly one")
    assert "needs Google Colab" in _refusal(section4, {"USE_BYOD": True}, colab=False)


def test_wrong_suffix_bad_content_and_contract_violations_are_refused_by_file_name(section4, tmp_path):
    cases = {
        "notes.txt": (b"hello\n", "must be .csv, .json or .jsonl"),
        "housing.csv": (b"longitude,latitude,median_house_value\n1,2,3\n", "CSV is missing columns"),
        "broken.json": (b"{not json", "Expecting property name"),
        "rare.csv": (_csv(_records({"category a": 40, "category b": 40, "category rare": 1})), "'category rare': 1"),
        "long.csv": (_csv(_records({"category a": 30, "category b": 30}, words=2700)), "tokens"),
    }
    for name, (payload, rule) in cases.items():
        path = tmp_path / name
        path.write_bytes(payload)
        line = _refusal(section4, {"USE_BYOD": True, "BYOD_PATH": str(path)})
        assert line.startswith(f"Refused: {name}: ") and rule in line, line
        assert line.endswith("Fix the file, then select this cell and choose Runtime → Run after."), line


def test_a_refusal_removes_the_earlier_runs_data_and_results(section4, tmp_path):
    good = tmp_path / "good.csv"
    good.write_bytes(_csv(_records({"category a": 10, "category b": 10})))
    ns, _out = section4({"USE_BYOD": True, "BYOD_PATH": str(good)})
    assert ns["train_records"] and Path(f"outputs/{STEM}_train.csv").is_file()
    Path(f"outputs/{STEM}_adapter").mkdir()
    Path(f"outputs/{STEM}_result.json").write_text("{}", encoding="utf-8")
    bad = tmp_path / "bad.csv"
    bad.write_bytes(b"a,b\n1,2\n")
    _refusal(section4, {"USE_BYOD": True, "BYOD_PATH": str(bad)})
    assert not list(Path("outputs").glob(f"{STEM}_*"))
    for name in ("records", "splits", "train_records", "val_records", "test_records", "labels", "token_budget", "data_source"):
        assert name not in ns, name


def test_a_genuine_failure_on_the_sample_path_still_raises_unchanged(section4):
    def broken(*args, **kwargs):
        raise ValueError("internal sample check failed")

    section4.ns["fetch_corpus"] = broken
    with pytest.raises(ValueError, match="internal sample check failed") as info:
        section4({"USE_BYOD": False})
    assert not getattr(info.value, "dimer_refusal", False)


def test_the_accepted_byod_file_still_runs_the_cell(section4, tmp_path):
    path = tmp_path / "local.csv"
    path.write_bytes(_csv(_records({"category a": 6, "category b": 6})))
    ns, out = section4({"USE_BYOD": True, "BYOD_PATH": str(path)}, colab=False)
    assert ns["data_source"] == "BYOD (local.csv)" and len(ns["train_records"]) == 8 and "'token_budget'" in out


# ---- the catch is narrow: only bad input is refused; an internal TypeError/OSError keeps its traceback ------------


def test_the_byod_catch_is_narrow_in_the_notebook_source():
    src = CELLS[SEC4].split("write_dataset_csv(", 1)[0]  # the BYOD try block; the probes below it are a separate demo
    assert "except (TypeError" not in src and "except TypeError" not in src
    assert src.count("except OSError as exc:") == 1
    assert re.search(r"try:\n            records = load_byod_dataset\(byod_path\)\n        except OSError as exc:\n", src)
    assert "\nexcept ValueError as exc:\n    if not USE_BYOD:\n        raise" in src


def test_an_unreadable_byod_file_is_still_one_refused_line(section4, tmp_path):
    path = tmp_path / "locked.csv"
    path.write_bytes(_csv(_records({"category a": 6, "category b": 6})))

    def unreadable(_path):
        raise PermissionError(13, "Permission denied", str(_path))

    section4.ns["load_byod_dataset"] = unreadable
    line = _refusal(section4, {"USE_BYOD": True, "BYOD_PATH": str(path)})
    assert line.startswith("Refused: locked.csv: the file cannot be read (PermissionError: "), line
    assert line.endswith("Fix the file, then select this cell and choose Runtime → Run after."), line


@pytest.mark.parametrize("error", [TypeError("internal helper bug"), OSError("internal I/O bug")])
def test_an_internal_error_after_the_loader_keeps_its_traceback(section4, tmp_path, error):
    import traceback

    path = tmp_path / "local.csv"
    path.write_bytes(_csv(_records({"category a": 6, "category b": 6})))

    def broken_token_count(text, labels, *, hypothesis_template):
        raise error

    section4.ns["pipe"].count_pair_tokens = broken_token_count
    try:
        with pytest.raises(type(error)) as info:
            section4({"USE_BYOD": True, "BYOD_PATH": str(path)})
    finally:
        del section4.ns["pipe"].count_pair_tokens
    assert info.value is error and not getattr(info.value, "dimer_refusal", False)
    rendered = "".join(traceback.format_exception(info.value))
    assert rendered.startswith("Traceback") and "broken_token_count" in rendered and "Refused:" not in rendered


def test_a_typeerror_inside_the_loader_is_not_turned_into_a_refusal(section4, tmp_path):
    path = tmp_path / "local.csv"
    path.write_bytes(_csv(_records({"category a": 6, "category b": 6})))

    def buggy_loader(_path):
        raise TypeError("loader bug")

    section4.ns["load_byod_dataset"] = buggy_loader
    with pytest.raises(TypeError, match="loader bug") as info:
        section4({"USE_BYOD": True, "BYOD_PATH": str(path)})
    assert not getattr(info.value, "dimer_refusal", False)


# ---- the router: a refusal crosses the worker boundary as one line; a genuine error keeps its traceback -----


def _router_namespace() -> dict:
    """The router cell's definitions (everything before it starts the worker), with a stand-in IPython."""
    src = CELLS[ROUTER]
    head = src[: src.index("\nif SKIP_INSTALL:")]
    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: None
    saved = sys.modules.get("IPython")
    sys.modules["IPython"] = ipython
    try:
        ns: dict = {"__name__": "__main__", "os": os, "subprocess": subprocess, "sys": sys}
        exec(compile(head, f"<cell {ROUTER}>", "exec"), ns)
    finally:
        if saved is None:
            sys.modules.pop("IPython", None)
        else:
            sys.modules["IPython"] = saved
    return ns


class _Queued:
    def __init__(self, messages):
        self.messages = list(messages)
        self.sent = []

    def recv(self):
        return self.messages.pop(0)

    def send(self, message):
        self.sent.append(message)


def _stub_runtime(ns, messages):
    runtime = object.__new__(ns["IsolatedRuntime"])
    runtime._recv, runtime._send, runtime._display = _Queued(messages), _Queued([]), lambda *a, **k: None
    return runtime


def test_router_carries_the_refusal_channel():
    src = CELLS[ROUTER]
    assert 'if getattr(exc, "dimer_refusal", False):' in src and '_send.send(("refusal", str(exc)))' in src
    assert 'elif kind == "refusal":' in src and "class IsolatedCellRefusal(Exception):" in src


def test_router_kernel_shows_a_refusal_as_one_line_and_an_error_with_its_traceback(capsys):
    ns = _router_namespace()
    with pytest.raises(ns["IsolatedCellRefusal"]) as info:
        _stub_runtime(ns, [("refusal", "housing.csv: CSV is missing columns ['id', 'label', 'text']")]).run("cell")
    assert info.value._render_traceback_() == ["Refused: housing.csv: CSV is missing columns ['id', 'label', 'text']"]
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""
    trace = "Traceback (most recent call last):\n  File x\nZeroDivisionError: division by zero\n"
    with pytest.raises(ns["IsolatedCellError"], match="ZeroDivisionError"):
        _stub_runtime(ns, [("error", trace, "ZeroDivisionError: division by zero")]).run("cell")
    assert capsys.readouterr().err == trace


@pytest.mark.skipif(sys.platform == "win32", reason="the router's worker uses pass_fds and POSIX sessions")
def test_real_worker_shows_the_section4_refusal_as_one_line_and_keeps_running(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    ns = _router_namespace()
    runtime = ns["IsolatedRuntime"](sys.executable, display=lambda *a, **k: None)
    try:
        runtime.run("import os")
        for index in MODULE_CELLS:
            runtime.run(CELLS[index])
        runtime.run(PIPE_STANDIN)
        bad = tmp_path / "housing.csv"
        bad.write_bytes(b"longitude,latitude,median_house_value\n1,2,3\n")
        capsys.readouterr()
        with pytest.raises(ns["IsolatedCellRefusal"]) as info:
            runtime.run(_section4_source({"USE_BYOD": True, "BYOD_PATH": str(bad)}))
        lines = info.value._render_traceback_()
        captured = capsys.readouterr()
        assert len(lines) == 1 and lines[0].startswith("Refused: housing.csv: CSV is missing columns"), lines
        assert "Traceback" not in captured.out + captured.err, captured
        runtime.run("print('still running', 'train_records' in globals())")
        assert capsys.readouterr().out == "still running False\n"
        with pytest.raises(ns["IsolatedCellError"]):
            runtime.run("1 / 0")
        assert "Traceback" in capsys.readouterr().err
    finally:
        runtime.close()
