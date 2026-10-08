#!/usr/bin/env python3
"""Check release patch identity; optionally reapply both series from a pinned local clone."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PIN = "cb2ebf0540f42604e2759b2ddef497861e928248"


def run(*args, **kwargs):
    return subprocess.run(args, check=True, text=True, capture_output=True, **kwargs).stdout


def manifest(root):
    files = []
    for p in sorted(root.rglob("*")):
        if p.is_symlink():
            raise RuntimeError(f"unexpected symlink: {p.relative_to(root)}")
        if p.is_file():
            files.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root)}\n")
    return len(files), hashlib.sha256("".join(files).encode()).hexdigest()


def clean_patch(source, target):
    for extra in [("--dry-run",), ()]:
        out = run("patch", "-p0", "--forward", "--batch", "--fuzz=0", *extra,
                  f"--input={source}", cwd=target)
        if any(word in out.lower() for word in ("fuzz", "offset")):
            raise RuntimeError(out)
    if any(p.suffix in (".orig", ".rej") for p in target.rglob("*")):
        raise RuntimeError("patch left a .orig or .rej file")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-patches", type=Path, required=True,
                        help="read-only directory containing the staged release's 86 patches")
    parser.add_argument("--upstream", type=Path,
                        help="read-only local TensorFold Git clone containing the pinned v0.6.6 commit")
    args = parser.parse_args()
    release = args.release_patches.resolve()
    apply = str(ROOT / "scripts/apply-patches.sh")
    series = run(apply, "--list").splitlines()
    assert len(series) == 86
    assert series == sorted(series, key=lambda name: Path(name).name)
    assert {p.name for p in release.glob("*.patch")} == {Path(p).name for p in series}
    for name in series:
        assert (ROOT / name).read_bytes() == (release / Path(name).name).read_bytes(), name
    print("PASS: all 86 default patches byte-identical to the staged release, same global order")
    print("Recipe label:", run(str(ROOT / "scripts/prepare.sh"), "--label").strip())
    if not args.upstream:
        print("NOT RUN: fresh apply (supply --upstream)")
        return
    upstream = args.upstream.resolve()
    assert run("git", "-C", str(upstream), "rev-parse", "v0.6.6^{commit}").strip() == PIN
    with tempfile.TemporaryDirectory(prefix=".apply-check-", dir=ROOT) as work:
        scratch = Path(work)
        archive = scratch / "base.tar"
        run("git", "-C", str(upstream), "archive", f"--output={archive}", PIN)
        current, baseline = scratch / "current", scratch / "baseline"
        current.mkdir()
        baseline.mkdir()
        for target in (current, baseline):
            run("tar", "-xf", str(archive), "-C", str(target))
        print(run(apply, str(current)).strip())
        for name in series:
            clean_patch(release / Path(name).name, baseline / "src")
        run("diff", "-qr", str(baseline), str(current))
        for p in baseline.rglob("*"):
            assert p.stat().st_mode == (current / p.relative_to(baseline)).stat().st_mode
        for scope in (".", "src/tensorfold"):
            count, digest = manifest(current / scope)
            print(f"PASS: {scope}: {count} files; manifest SHA-256 {digest}")


if __name__ == "__main__":
    main()
