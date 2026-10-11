import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent


def main():
    scripts = [
        ROOT_DIR / "Day2" / "model_creation.py",
        ROOT_DIR / "Day1" / "hands_on_starter.py",
    ]
    output_sections = []

    for script in scripts:
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=script.parent,
            check=True,
            capture_output=True,
            text=True,
        )
        output_sections.append(f"=== {script.name} ===\n{result.stdout}")
        if result.stderr:
            output_sections.append(f"[stderr from {script.name}]\n{result.stderr}")

    output_file = ROOT_DIR / "cobined_output.txt"
    combined_output = "\n".join(output_sections)
    output_file.write_text(combined_output, encoding="utf-8")
    print(combined_output)
    print(f"Combined output saved to {output_file}")


if __name__ == "__main__":
    main()
