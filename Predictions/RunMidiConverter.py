import subprocess
import sys
from pathlib import Path
import shutil


def main():
    # The folder where this script is located
    folder = Path(__file__).resolve().parent

    # Path to the exe (relative to this script or adjust as needed)
    exe_path = (
        Path("DNA_App") / "DNAConsole" / "bin" / "Debug" / "net8.0" / "DNAConsole.exe"
    ).resolve()

    if not exe_path.is_file():
        print(f"[ERROR] Executable not found at: {exe_path}")
        sys.exit(1)

    # Collect all JSON files in this folder
    json_files = list(folder.glob("*.json"))

    if not json_files:
        print(f"[INFO] No JSON files found in {folder}")
        sys.exit(0)

    for json_file in json_files:
        print(f"\nRunning DNAConsole.exe with --dna2midi on: {json_file.name}")

        try:
            result = subprocess.run(
                [str(exe_path), "--dna2midi", str(json_file)],
                check=True,
                capture_output=True,
                text=True,
            )

            print(" STDOUT ")
            print(result.stdout)
            print(" STDERR ")
            print(result.stderr)

            # Default output filename used by DNAConsole
            default_midi = folder / "expansion0.mid"

            # Target filename based on input JSON
            target_midi = folder / f"{json_file.stem}.mid"

            if default_midi.exists():
                shutil.move(default_midi, target_midi)
                print(f"[SUCCESS] Created {target_midi.name}")
            else:
                print(f"[WARNING] Expected output {default_midi} not found!")

        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Execution failed for {json_file.name} with return code {e.returncode}")
            print("=== STDOUT ===")
            print(e.output)
            print("=== STDERR ===")
            print(e.stderr)


if __name__ == "__main__":
    main()
