"""CLI help/validation, HPC script syntax and dry run, and a scan for sensitive values."""
import os
import re
import shutil
import subprocess
import sys

import pytest

from conftest import REPO

MODULES = ["src.data.build_ben_s2_metadata", "src.data.validate_ben_s2_metadata",
           "src.data.recommend_ben_s2_split_candidates", "src.data.create_ben_s2_first_split",
           "src.data.recommend_ben_s2_label_aware_candidates", "src.data.create_ben_s2_labelhard_split",
           "src.data.export_ben_s2_subset", "src.data.check_ben_s2_loader", "src.training.train_baseline"]
BASH = shutil.which("bash")


def _run(args, **kw):
    return subprocess.run(args, cwd=REPO, capture_output=True, text=True, timeout=300, **kw)


@pytest.mark.parametrize("module", MODULES)
def test_cli_help(module):
    assert _run([sys.executable, "-m", module, "--help"]).returncode == 0


@pytest.mark.parametrize("extra", [
    [],                                                          # --pretrained is required
    ["--pretrained", "false", "--pretrained_weights", "w.pth"],  # weights need --pretrained true
    ["--pretrained", "false", "--epochs", "0"],
])
def test_train_cli_rejects_invalid_arguments(extra, tmp_path):
    res = _run([sys.executable, "-m", "src.training.train_baseline", "--mode", "head_only",
                "--split_dir", str(tmp_path), "--output_dir", str(tmp_path / "o"), *extra])
    assert res.returncode == 2


NCI_SCRIPTS = sorted((REPO / "nci").glob("*.pbs")) + sorted((REPO / "nci").glob("*.sh"))


@pytest.mark.skipif(BASH is None, reason="bash not available")
@pytest.mark.parametrize("script", NCI_SCRIPTS, ids=lambda p: p.name)
def test_nci_shell_syntax(script):
    assert _run([BASH, "-n", str(script)]).returncode == 0


def _stub_env(tmp_path):
    """Fake `module` and `python` so train_common.sh can be dry-run without Gadi."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "module").write_text("#!/usr/bin/env bash\nexit 0\n")
    (bin_dir / "python").write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$ARGS_OUT"\n')
    for f in bin_dir.iterdir():
        f.chmod(0o755)
    venv = tmp_path / "venv" / "bin"
    venv.mkdir(parents=True)
    (venv / "activate").write_text("")
    return bin_dir, tmp_path / "venv"


@pytest.mark.skipif(BASH is None or os.name == "nt" and not shutil.which("cygpath"), reason="needs a POSIX bash")
def test_train_common_requires_variables_and_builds_command(tmp_path):
    bin_dir, venv = _stub_env(tmp_path)
    to_posix = (lambda p: subprocess.run(["cygpath", "-u", str(p)], capture_output=True, text=True).stdout.strip()) \
        if os.name == "nt" else str
    base = {**os.environ, "PATH": to_posix(bin_dir) + os.pathsep.replace(";", ":") + os.environ.get("PATH", ""),
            "MODE": "pooled_lora", "LR": "0.0001", "GLUE_ROOT": to_posix(REPO), "SPLIT_DIR": "/s",
            "DATA_ROOT": "/d", "OUTPUT_DIR": to_posix(tmp_path / "out"), "VENV": to_posix(venv),
            "ARGS_OUT": to_posix(tmp_path / "args.txt")}
    if os.name == "nt":
        base["PATH"] = to_posix(bin_dir) + ":" + "/usr/bin:/bin"
    script = to_posix(REPO / "nci" / "train_common.sh")

    missing = _run([BASH, script], env={k: v for k, v in base.items() if k != "PRETRAINED"})
    assert missing.returncode != 0 and "PRETRAINED" in missing.stderr

    ok = _run([BASH, script], env={**base, "PRETRAINED": "true", "PRETRAINED_WEIGHTS": "/w/resnet50.pth"})
    assert ok.returncode == 0, ok.stderr
    args = (tmp_path / "args.txt").read_text().split("\n")
    assert args[:2] == ["-m", "src.training.train_baseline"]
    for flag, value in [("--mode", "pooled_lora"), ("--pretrained", "true"), ("--pretrained_weights", "/w/resnet50.pth"),
                        ("--lr", "0.0001"), ("--split_dir", "/s")]:
        assert args[args.index(flag) + 1] == value


SENSITIVE = {
    "real NCI scratch path": re.compile(r"/scratch/[a-z]{1,3}\d{1,3}/[a-z]{2}\d{4}"),
    "Windows user profile path": re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+\w+", re.I),
    "private IPv4": re.compile(r"\b(10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+)\b"),
    "token": re.compile(r"(gh[pousr]_[A-Za-z0-9]{36}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY)"),
    "literal password": re.compile(r"(?i)password\s*[:=]\s*['\"][^'\"<]{3,}"),
}


def test_tracked_files_contain_no_sensitive_values():
    files = _run(["git", "ls-files"]).stdout.split()
    if not files:
        pytest.skip("not a git checkout")
    hits = []
    for name in files:
        path = REPO / name
        if not path.is_file() or path.suffix in {".pt", ".pth", ".tif"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for label, rx in SENSITIVE.items():
            if name.endswith("test_cli_and_scripts.py"):
                continue
            if rx.search(text):
                hits.append((name, label))
    assert hits == []
