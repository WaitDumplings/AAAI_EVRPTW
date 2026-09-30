from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import pytest


CODE_ROOT = Path(__file__).resolve().parents[1]
FAKE_RUNNER = r"""
import json
import os
from pathlib import Path
import sys
import time

record_path = Path(os.environ["LAUNCHER_TEST_RECORD"])
release = Path(os.environ["LAUNCHER_TEST_RELEASE"])
record = {
    "argv": sys.argv[1:],
    "pid": os.getpid(),
    "session": os.getsid(0),
    "stdin": os.readlink("/proc/self/fd/0"),
}
temporary = record_path.with_suffix(".tmp")
temporary.write_text(json.dumps(record))
temporary.replace(record_path)
print("fake runner started", flush=True)
deadline = time.monotonic() + 15
while not release.exists() and time.monotonic() < deadline:
    time.sleep(0.02)
print("fake runner finished", flush=True)
record_path.with_suffix(".finished").touch()
"""


def wait_for_file(path: Path, timeout: float = 5) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            pytest.fail(f"Background runner did not create {path}")
        time.sleep(0.02)


@dataclass
class Checkout:
    root: Path
    outside: Path
    env: dict[str, str]
    record_path: Path
    release: Path

    def dataset(self, problem: str, scale: int) -> Path:
        return self.root.parent / "AAAI_Dataset" / "test_release" / problem / "test" / f"Cus{scale}"

    def output(self, problem: str, scale: int) -> Path:
        return self.root / "results" / "gurobi" / problem / "test" / f"Cus{scale}"

    def run(self, *args: str, shell: bool = False) -> subprocess.CompletedProcess[str]:
        if shell:
            command = ["bash", str(self.root / "scripts" / "run_gurobi_test.sh")]
        else:
            command = [sys.executable, "-B", str(self.root / "scripts" / "run_gurobi_test.py")]
        return subprocess.run(
            [*command, *args], cwd=self.outside, env=self.env,
            text=True, capture_output=True, timeout=5,
        )

    def record(self) -> dict:
        wait_for_file(self.record_path)
        return json.loads(self.record_path.read_text())


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / "checkout with spaces" / "CaliRoute"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    for name in ("run_gurobi_test.py", "run_gurobi_test.sh"):
        shutil.copy2(CODE_ROOT / "scripts" / name, scripts / name)
    helper = Path("EVRPTW_Benchmark/Exact/Gurobi_Solver/resume.py")
    (root / helper).parent.mkdir(parents=True)
    shutil.copy2(CODE_ROOT / helper, root / helper)
    outside = tmp_path / "outside checkout"
    outside.mkdir()
    stubs = tmp_path / "stub modules"
    stubs.mkdir()
    (stubs / "gurobipy.py").write_text("# No Gurobi dependency is needed by the fake runner.\n")
    binaries = tmp_path / "python bin"
    binaries.mkdir()
    (binaries / "python").symlink_to(sys.executable)
    (binaries / "python3").symlink_to(sys.executable)
    record_path = tmp_path / "runner.json"
    release = tmp_path / "release runner"
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(stubs),
        PATH=str(binaries) + os.pathsep + os.environ["PATH"],
        LAUNCHER_TEST_RECORD=str(record_path),
        LAUNCHER_TEST_RELEASE=str(release),
    )
    instance = Checkout(root, outside, env, record_path, release)
    for problem in ("cvrp", "vrptw", "evrptw"):
        runner = root / "EVRPTW_Benchmark" / "Exact" / "Gurobi_Solver" / problem.upper() / "run_gurobi.py"
        runner.parent.mkdir(parents=True)
        runner.write_text(FAKE_RUNNER)
        for scale in (15, 50, 100):
            dataset = instance.dataset(problem, scale)
            dataset.mkdir(parents=True)
            # Deliberately not a pickle: the launcher must only inspect metadata.
            (dataset / "instances.pkl").write_bytes(b"fake test bundle")
            (dataset / "metadata.json").write_text(json.dumps({
                "num_instances": 1000, "split": "test", "num_customers": scale,
            }))
    yield instance
    release.touch()
    if record_path.exists():
        deadline = time.monotonic() + 5
        while not record_path.with_suffix(".finished").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not record_path.with_suffix(".finished").exists():
            try:
                os.kill(json.loads(record_path.read_text())["pid"], signal.SIGTERM)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize("problem", ["cvrp", "vrptw", "evrptw"])
@pytest.mark.parametrize("scale", [15, 50, 100])
def test_launches_complete_test_bundle_with_fixed_budget(checkout, problem, scale):
    result = checkout.run(problem, str(scale))
    assert result.returncode == 0, result.stderr
    record = checkout.record()
    argv = record["argv"]
    expected = {
        "--dataset_path": str(checkout.dataset(problem, scale) / "instances.pkl"),
        "--save_path": str(checkout.output(problem, scale)),
        "--workers": "30",
        "--threads": "1",
        "--time_limit_s": "7200",
        "--mip_gap": "0",
        "--checkpoints_s": "60,300,900,3600,7200",
    }
    for option, value in expected.items():
        actual = argv[argv.index(option) + 1]
        if option in {"--time_limit_s", "--mip_gap"}:
            assert float(actual) == float(value)
        else:
            assert actual == value
    assert {"--skip_completed", "--verbose", "--save_traceback"}.issubset(argv)
    assert not {"--start_index", "--end_index", "--limit"}.intersection(argv)
    if problem == "evrptw":
        assert argv[argv.index("--cs_copies") + 1] == {15: "1", 50: "2", 100: "2"}[scale]
        assert argv[argv.index("--reference_split") + 1] == "test"
        assert "--no-tie_break_vehicle_count" in argv
    else:
        assert "--cs_copies" not in argv
        assert "--no-tie_break_vehicle_count" not in argv
    assert record["session"] == record["pid"]
    assert record["stdin"] == "/dev/null"
    assert not checkout.release.exists()
    assert int((checkout.output(problem, scale) / "launcher.pid").read_text()) == record["pid"]
    logs = list((checkout.output(problem, scale) / "logs").glob("*.log"))
    assert len(logs) == 1
    assert str(logs[0]) in result.stdout
    assert str(record["pid"]) in result.stdout


def test_shell_wrapper_works_outside_checkout_and_rejects_duplicate_run(checkout):
    result = checkout.run("vrptw", "50", shell=True)
    assert result.returncode == 0, result.stderr
    first = checkout.record()
    duplicate = checkout.run("vrptw", "50", shell=True)
    assert duplicate.returncode != 0
    assert checkout.record() == first
    assert len(list((checkout.output("vrptw", 50) / "logs").glob("*.log"))) == 1


@pytest.mark.parametrize("args", [
    ("tsp", "15"), ("cvrp", "20"), ("cvrp",), ("cvrp", "15", "extra"),
])
def test_rejects_arguments_outside_two_supported_controls(checkout, args):
    result = checkout.run(*args)
    assert result.returncode != 0
    assert not checkout.record_path.exists()


@pytest.mark.parametrize("invalid", [
    {"num_instances": 999}, {"split": "val"}, {"num_customers": 50},
])
def test_rejects_unexpected_test_bundle_metadata(checkout, invalid):
    metadata = checkout.dataset("cvrp", 15) / "metadata.json"
    payload = json.loads(metadata.read_text())
    payload.update(invalid)
    metadata.write_text(json.dumps(payload))
    result = checkout.run("cvrp", "15")
    assert result.returncode != 0
    assert not checkout.record_path.exists()


def test_rejects_missing_test_bundle(checkout):
    (checkout.dataset("evrptw", 100) / "instances.pkl").unlink()
    result = checkout.run("evrptw", "100")
    assert result.returncode != 0
    assert not checkout.record_path.exists()


@pytest.mark.parametrize("problem", ["cvrp", "vrptw", "evrptw"])
def test_shell_reports_existing_results_without_modifying_summary(checkout, problem):
    output = checkout.output(problem, 15)
    output.mkdir(parents=True)
    summary = output / "gurobi_summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["instance_id", "status_name", "feasible"])
        writer.writeheader()
        writer.writerows([
            {"instance_id": "finished_000000", "status_name": "OPTIMAL", "feasible": True},
            {"instance_id": "limited_000001", "status_name": "TIME_LIMIT", "feasible": False},
            {"instance_id": "retry_000002", "status_name": "ERROR", "feasible": False},
            {"instance_id": "stopped_000003", "status_name": "INTERRUPTED", "feasible": False},
            # A repeated record counts once and the latest status determines completion.
            {"instance_id": "finished_000000", "status_name": "OPTIMAL", "feasible": True},
        ])
    original = summary.read_bytes()
    result = checkout.run(problem, "15", shell=True)
    assert result.returncode == 0, result.stderr
    assert "--skip_completed" in checkout.record()["argv"]
    log = next((output / "logs").glob("*.log")).read_text()
    for message in (result.stdout, log):
        assert "Resume: enabled" in message
        assert f"summary={summary}" in message
        assert "recorded_completed=2" in message
        assert "exact skipped/pending counts appear in the log" in message
    assert summary.read_bytes() == original
