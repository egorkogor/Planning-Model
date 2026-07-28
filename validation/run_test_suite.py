from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_ROOTS = (ROOT / "validation", ROOT / "tests")


@dataclass(frozen=True)
class FileResult:
    path: Path
    returncode: int
    passed: int
    timed_out: bool
    log_path: Path


def discover_test_files() -> list[Path]:
    files: list[Path] = []
    for test_root in TEST_ROOTS:
        files.extend(sorted(test_root.glob("test_*.py")))
    return files


def _terminate_process_tree(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            proc.terminate()
        else:
            os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            if os.name == "nt":
                proc.kill()
            else:
                os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=5)


def _count_junit_tests(path: Path) -> int:
    if not path.is_file():
        return 0
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    passed = 0
    for suite in suites:
        total = int(suite.attrib.get("tests", "0"))
        failed = int(suite.attrib.get("failures", "0"))
        errors = int(suite.attrib.get("errors", "0"))
        skipped = int(suite.attrib.get("skipped", "0"))
        passed += total - failed - errors - skipped
    return passed


def run_test_file(path: Path, *, timeout_seconds: int, work_dir: Path, log_dir: Path) -> FileResult:
    rel = path.relative_to(ROOT)
    safe_name = "__".join(rel.parts).replace(".py", "")
    base_temp = work_dir / safe_name
    junit_path = work_dir / f"{safe_name}.xml"
    log_path = log_dir / f"{safe_name}.log"
    shutil.rmtree(base_temp, ignore_errors=True)

    env = dict(os.environ)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        str(rel),
        f"--basetemp={base_temp}",
        f"--junitxml={junit_path}",
        "-p",
        "no:cacheprovider",
    ]
    popen_kwargs: dict[str, object] = {
        "cwd": ROOT,
        "env": env,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    else:
        popen_kwargs["start_new_session"] = True

    timed_out = False
    with log_path.open("wb") as log_handle:
        proc = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            **popen_kwargs,  # type: ignore[arg-type]
        )
        try:
            proc.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_process_tree(proc)

    return FileResult(
        path=rel,
        returncode=proc.returncode if proc.returncode is not None else 1,
        passed=_count_junit_tests(junit_path),
        timed_out=timed_out,
        log_path=log_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run repository tests in isolated subprocesses.")
    parser.add_argument("--timeout-per-file", type=int, default=120)
    parser.add_argument("--keep-logs", action="store_true")
    parser.add_argument("--log-dir", type=Path)
    args = parser.parse_args()
    if args.timeout_per_file < 1:
        parser.error("--timeout-per-file must be positive")

    test_files = discover_test_files()
    if not test_files:
        print("FAIL: no test files discovered", file=sys.stderr)
        return 2

    auto_log_dir = args.log_dir is None
    log_dir = args.log_dir or Path(tempfile.mkdtemp(prefix="planner-test-logs-"))
    log_dir = log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="planner-pytest-basetemp-"))

    results: list[FileResult] = []
    try:
        for index, path in enumerate(test_files, start=1):
            rel = path.relative_to(ROOT)
            print(f"[{index:02d}/{len(test_files):02d}] {rel}", flush=True)
            result = run_test_file(
                path,
                timeout_seconds=args.timeout_per_file,
                work_dir=work_dir,
                log_dir=log_dir,
            )
            results.append(result)
            if result.timed_out:
                print(f"  TIMEOUT after {args.timeout_per_file}s; log: {result.log_path}", flush=True)
            elif result.returncode != 0:
                print(f"  FAIL (exit {result.returncode}); log: {result.log_path}", flush=True)
            else:
                print(f"  PASS ({result.passed} tests)", flush=True)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    failed = [result for result in results if result.returncode != 0 or result.timed_out]
    total_passed = sum(result.passed for result in results)
    if failed:
        print(f"FAIL: {len(failed)}/{len(results)} test files failed; {total_passed} tests passed", file=sys.stderr)
        print(f"Logs: {log_dir}", file=sys.stderr)
        return 1

    print(f"PASS: {len(results)} test files; {total_passed} tests passed")
    if args.keep_logs or not auto_log_dir:
        print(f"Logs: {log_dir}")
    else:
        shutil.rmtree(log_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
