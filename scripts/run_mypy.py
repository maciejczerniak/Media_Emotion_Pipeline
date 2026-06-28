import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    if sys.platform == "win32":
        venv_python = repo_root / ".venv" / "Scripts" / "python.exe"
    else:
        venv_python = repo_root / ".venv" / "bin" / "python"

    python = venv_python if venv_python.exists() else Path(sys.executable)
    return subprocess.call([str(python), "-m", "mypy", *sys.argv[1:]], cwd=repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
