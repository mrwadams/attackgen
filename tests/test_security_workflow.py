"""The security workflow's summary job reports every other job's real result."""

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "security.yml"
SUMMARY_JOB = "security-summary"


def _jobs():
    return yaml.safe_load(WORKFLOW.read_text())["jobs"]


def test_summary_needs_every_other_job():
    jobs = _jobs()
    assert set(jobs[SUMMARY_JOB]["needs"]) == set(jobs) - {SUMMARY_JOB}


def _summary_step():
    (step,) = _jobs()[SUMMARY_JOB]["steps"]
    return step


def _run_summary(tmp_path, results):
    """Run the summary step as GitHub would, given each job's result."""
    summary = tmp_path / "summary.md"
    needs = {job: {"result": result, "outputs": {}} for job, result in results.items()}
    env = {**os.environ, "NEEDS": json.dumps(needs), "GITHUB_STEP_SUMMARY": str(summary)}
    subprocess.run(["bash", "-e", "-c", _summary_step()["run"]], env=env, check=True)
    return summary.read_text().splitlines()


@pytest.mark.parametrize(
    "result, line",
    [
        ("success", "- ✅ python-security: success"),
        ("failure", "- ❌ python-security: failure"),
        ("skipped", "- ⏭️ python-security: skipped"),
        ("cancelled", "- ⚠️ python-security: cancelled"),
    ],
)
def test_summary_marks_each_result(tmp_path, result, line):
    assert line in _run_summary(tmp_path, {"python-security": result})


def test_summary_lists_every_job_it_receives(tmp_path):
    lines = _run_summary(
        tmp_path,
        {"python-security": "success", "docker-security": "skipped", "dependency-review": "failure"},
    )
    assert "- ✅ python-security: success" in lines
    assert "- ⏭️ docker-security: skipped" in lines
    assert "- ❌ dependency-review: failure" in lines


def test_summary_step_reads_results_from_needs():
    assert _summary_step()["env"]["NEEDS"] == "${{ toJSON(needs) }}"
