import argparse
import subprocess
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Run DNAConsole.exe with a path argument.")
    parser.add_argument("path", help="The path to midi + text files")
    args = parser.parse_args()

    # Get the absolute path to the .exe
    exe_path = os.path.abspath(os.path.join("DNA_App", "DNAConsole", "bin", "Debug", "net8.0", "DNAConsole.exe"))

    if not os.path.isfile(exe_path):
        print(f"Executable not found at: {exe_path}")
        sys.exit(1)

    # Call the .exe with the provided path argument
    try:
        subprocess.run([exe_path, args.path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Execution failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()