import os
import subprocess
import tempfile
import time
from pathlib import Path
from dataclasses import dataclass

from dotenv import load_dotenv
from github import Github
from git import Repo

from openai import OpenAI

# ---------- CONFIG ----------

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO")  # e.g. "username/myrepo"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_BRANCH = os.getenv("BASE_BRANCH", "main")
MODEL_NAME = "gpt-4.1-mini"  # adjust as needed

client = OpenAI(api_key=OPENAI_API_KEY)

repo_path = Path(__file__).resolve().parent

# ---------- DATA STRUCTS ----------

@dataclass
class TestResult:
    success: bool
    output: str
    error: str


# ---------- STEP 1: RUN TESTS ----------

def run_pytest() -> TestResult:
    print("▶ Running pytest...")
    proc = subprocess.Popen(
        ["pytest", "-q"],
        cwd=repo_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = proc.communicate()
    success = proc.returncode == 0
    print("✅ Tests passed" if success else "❌ Tests failed")
    return TestResult(success=success, output=stdout, error=stderr)


# ---------- STEP 2: COLLECT CONTEXT ----------

def extract_error_snippet(test_result: TestResult, max_lines: int = 80) -> str:
    """
    For POC, we just send last N lines of stderr to the model.
    Later: parse traceback to open files and add more code context.
    """
    err_lines = test_result.error.strip().splitlines()
    snippet = "\n".join(err_lines[-max_lines:])
    return snippet


def get_repo_code_snapshot(max_files: int = 10, max_chars_per_file: int = 2000) -> str:
    """
    POC: send a small snapshot of .py files for context.
    Later: only files mentioned in traceback.
    """
    parts = []
    count = 0
    for py_file in repo_path.rglob("*.py"):
        if ".venv" in str(py_file) or "site-packages" in str(py_file):
            continue
        if count >= max_files:
            break
        rel = py_file.relative_to(repo_path)
        try:
            text = py_file.read_text(encoding="utf-8")
        except Exception:
            continue
        text = text[:max_chars_per_file]
        parts.append(f"### FILE: {rel}\n{text}\n")
        count += 1
    return "\n\n".join(parts)


# ---------- STEP 3: ASK LLM FOR PATCH (UNIFIED DIFF) ----------

def ask_llm_for_patch(error_snippet: str, code_snapshot: str) -> str:
    """
    Ask the LLM to output a unified diff patch only.
    """
    system_prompt = """
You are an expert software engineer.
Given Python project files and a failing pytest traceback,
you must FIX THE BUG by returning a UNIX unified diff patch.

STRICT RULES:
- Output ONLY a unified diff starting with lines like: diff --git a/... b/...
- DO NOT write explanations, comments, or markdown.
- Keep changes minimal and targeted.
- Ensure syntax is valid and tests are more likely to pass.
"""
    user_prompt = f"""
The tests in this repository are failing. Here is the traceback:

<ERROR>
{error_snippet}
</ERROR>

Here is a snapshot of the repository code:

<CODE>
{code_snapshot}
</CODE>

Return ONLY the unified diff patch to fix the problem.
"""

    print("🤖 Calling LLM for patch...")
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0.2,
    )

    patch_text = resp.choices[0].message.content.strip()
    print("📦 Received patch from LLM (truncated preview):")
    print("\n".join(patch_text.splitlines()[:15]), "\n...")
    return patch_text


# ---------- STEP 4: APPLY PATCH ----------

def apply_patch(patch_text: str) -> bool:
    """
    Apply unified diff using `patch` command.
    """
    print("🩹 Applying patch...")
    with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
        tmp.write(patch_text)
        tmp_path = tmp.name

    try:
        proc = subprocess.Popen(
            ["patch", "-p1", "-i", tmp_path],
            cwd=repo_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = proc.communicate()
        success = proc.returncode == 0
        print(stdout)
        if not success:
            print("❌ Failed to apply patch:")
            print(stderr)
        else:
            print("✅ Patch applied successfully")
        return success
    finally:
        os.remove(tmp_path)


# ---------- STEP 5: CREATE BRANCH, COMMIT, PUSH & PR ----------

def compute_confidence_score(before_failed: bool, after_passed: bool, lines_changed: int) -> float:
    """
    Very simple heuristic confidence for POC.
    """
    if not after_passed:
        return 0.0
    # smaller change = slightly higher confidence
    size_factor = 1 / (1 + lines_changed / 50)
    base = 0.7 if before_failed and after_passed else 0.3
    return round(base + 0.3 * size_factor, 2)


def get_changed_lines(repo: Repo) -> int:
    diff_text = repo.git.diff()
    return sum(1 for line in diff_text.splitlines() if line.startswith(("+", "-")))


def create_branch_and_pr(pr_title: str, pr_body: str) -> None:
    print("🔀 Creating branch, committing, pushing and opening PR...")
    repo = Repo(str(repo_path))

    # Make sure we are on BASE_BRANCH
    repo.git.checkout(BASE_BRANCH)
    repo.git.pull()

    branch_name = f"self-heal/fix-{int(time.time())}"
    repo.git.checkout("-b", branch_name)

    repo.git.add(A=True)
    repo.index.commit(pr_title)

    origin = repo.remote(name="origin")
    origin.push(branch_name)

    gh = Github(GITHUB_TOKEN)
    gh_repo = gh.get_repo(GITHUB_REPO)

    gh_pr = gh_repo.create_pull(
        title=pr_title,
        body=pr_body,
        head=branch_name,
        base=BASE_BRANCH,
    )

    print(f"✅ PR created: {gh_pr.html_url}")


# ---------- ORCHESTRATOR ----------

def main():
    print("===== SELF-HEALING PR AGENT (POC) =====")

    # 1. Run tests initially
    initial_result = run_pytest()
    if initial_result.success:
        print("✅ Tests already passing. No healing needed.")
        return

    error_snippet = extract_error_snippet(initial_result)
    code_snapshot = get_repo_code_snapshot()

    # 2. Get patch from LLM
    patch_text = ask_llm_for_patch(error_snippet, code_snapshot)

    # 3. Apply patch
    if not apply_patch(patch_text):
        print("💔 Stopping: patch failed to apply.")
        return

    # 4. Run tests again
    healed_result = run_pytest()

    # 5. If tests passed, create PR
    repo = Repo(str(repo_path))
    lines_changed = get_changed_lines(repo)
    confidence = compute_confidence_score(
        before_failed=not initial_result.success,
        after_passed=healed_result.success,
        lines_changed=lines_changed,
    )

    if healed_result.success:
        pr_title = f"Self-healing fix (confidence {confidence})"
        pr_body = f"""This pull request was generated automatically by the Self-Healing PR Agent.

- Initial tests: failing
- After patch: all tests passing ✅
- Lines changed: ~{lines_changed}
- Confidence score: {confidence}

Traceback (before fix):

{extract_error_snippet(initial_result, max_lines=40)}

Logs (after fix):

{healed_result.output[:1000]}
"""
        create_branch_and_pr(pr_title, pr_body)
    else:
        print("❌ Tests are still failing after patch. No PR created.")
        print("You can inspect the changes manually now.")


if __name__ == "__main__":
    main()
