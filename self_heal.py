import os
import subprocess
import time
import json
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

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in .env")

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


# ---------- STEP 3: ASK LLM FOR FIX (FULL FILE CONTENT) ----------

def ask_llm_for_fix(error_snippet: str, code_snapshot: str) -> tuple[str, str]:
    """
    Ask the LLM to return JSON with:
    {
      "filename": "relative/path/to/file.py",
      "new_content": "<full corrected file content>"
    }
    """
    system_prompt = """
You are an expert Python developer helping to fix failing pytest tests.

Given:
- A Python project snapshot
- The pytest traceback

You must:
1. Identify ONE most relevant Python source file that needs to be changed.
2. Return the COMPLETE corrected content of that file.

STRICT RULES:
- Respond ONLY with valid JSON, no comments, no markdown.
- JSON format must be exactly:
  {
    "filename": "relative/path/from/repo/root.py",
    "new_content": "full corrected file content here"
  }
- "filename" must refer to an existing .py file in the repository snapshot.
- "new_content" must be valid Python and include the entire file, not a diff.
"""
    user_prompt = f"""
The tests in this repository are failing. Here is the pytest traceback:

<ERROR>
{error_snippet}
</ERROR>

Here is a snapshot of the repository code:

<CODE>
{code_snapshot}
</CODE>

Return ONLY the JSON object as described. Do not add explanations.
"""

    print("🤖 Calling LLM for full-file fix...")
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0.2,
    )

    content = resp.choices[0].message.content.strip()
    print("📦 Raw LLM JSON (truncated):")
    print(content[:300], "...\n")

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        print("❌ Failed to parse LLM JSON:", e)
        raise

    filename = data["filename"]
    new_content = data["new_content"]
    return filename, new_content


# ---------- STEP 4: APPLY FIX (OVERWRITE FILE) ----------

def apply_fix(filename: str, new_content: str) -> bool:
    """
    Overwrite the specified file with new_content.
    """
    target_path = repo_path / filename
    print(f"📝 Applying fix to file: {filename}")

    if not target_path.exists():
        print(f"❌ File not found in repo: {filename}")
        return False

    try:
        old_content = target_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"❌ Failed to read original file: {e}")
        return False

    try:
        target_path.write_text(new_content, encoding="utf-8")
    except Exception as e:
        print(f"❌ Failed to write new file content: {e}")
        return False

    print("✅ File overwritten with new content")
    return True


# ---------- STEP 5: CONFIDENCE + GIT + PR ----------

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

    # 2. Get full-file fix from LLM
    try:
        filename, new_content = ask_llm_for_fix(error_snippet, code_snapshot)
    except Exception:
        print("💔 Stopping: LLM did not return valid JSON.")
        return

    # 3. Apply fix
    if not apply_fix(filename, new_content):
        print("💔 Stopping: failed to apply fix.")
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
        print("❌ Tests are still failing after fix. No PR created.")
        print("You can inspect the changes manually now.")


if __name__ == "__main__":
    main()
