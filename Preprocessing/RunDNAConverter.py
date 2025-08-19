import argparse
import glob
import shutil
import subprocess
import sys
from pathlib import Path


def unique_target(path: Path) -> Path:
    """
    If 'path' exists, append _1, _2, ... before the suffix to avoid collisions.
    """
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    i = 1
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def run_on_folder(exe_path: Path, folder: Path) -> bool:
    """
    Run the exe on a single folder. Returns True on success.
    """
    try:
        subprocess.run([str(exe_path), str(folder)], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Execution failed for {folder} with error: {e}")
        return False


def move_jsons_up(from_dir: Path, to_dir: Path) -> int:
    """
    Move all .json files from 'from_dir' to 'to_dir', resolving name collisions.
    Returns the count of moved files.
    """
    moved = 0
    for json_path_str in glob.glob(str(from_dir / "*.json")):
        src = Path(json_path_str)
        # Prefer prefixing with subfolder name to keep provenance
        target_name = f"{from_dir.name}__{src.name}"
        target = unique_target(to_dir / target_name)
        shutil.move(str(src), str(target))
        moved += 1
        print(f"Moved JSON: {src} -> {target}")
    return moved


def cleanup_mid_txt(folder: Path) -> int:
    """
    Delete .mid, .midi, and .txt files within 'folder' (non-recursive).
    Returns the count of deleted files.
    """
    patterns = ["*.mid", "*.midi", "*.txt"]
    deleted = 0
    for pattern in patterns:
        for file_path_str in glob.glob(str(folder / pattern)):
            p = Path(file_path_str)
            try:
                p.unlink()
                deleted += 1
            except OSError as e:
                print(f"Error deleting {p}: {e}")
    return deleted


def main():
    parser = argparse.ArgumentParser(description="Run DNAConsole.exe on subfolders, collect JSONs, and clean up.")
    parser.add_argument("path", help="Path containing chunk subfolders (or files)")
    args = parser.parse_args()

    root = Path(args.path).resolve()

    exe_path = (Path("DNA_App") / "DNAConsole" / "bin" / "Debug" / "net8.0" / "DNAConsole.exe").resolve()
    if not exe_path.is_file():
        print(f"Executable not found at: {exe_path}")
        sys.exit(1)

    # Gather immediate subdirectories (e.g., chunk_0000, chunk_0001, ...)
    subdirs = sorted([p for p in root.iterdir() if p.is_dir()])

    if not subdirs:
        # No subdirs -> process root like before
        print(f"No subfolders found under {root}. Running on the root folder.")
        if run_on_folder(exe_path, root):
            # Move JSONs produced in root to root (no-op), then clean up files
            moved = move_jsons_up(root, root)
            deleted = cleanup_mid_txt(root)
            print(f"Root processing complete. JSONs moved: {moved}, files deleted: {deleted}")
        else:
            sys.exit(1)
        return

    # Process each subfolder
    for sub in subdirs:
        print(f"=== Processing subfolder: {sub} ===")
        ok = run_on_folder(exe_path, sub)
        if not ok:
            print(f"Skipping cleanup for {sub} due to error.")
            continue

        # Move JSONs up to the root folder
        moved_jsons = move_jsons_up(sub, root)

        # Clean up mid/txt ONLY if run succeeded
        deleted_files = cleanup_mid_txt(sub)

        # Remove the entire subfolder if everything above succeeded
        # (Even if there were 0 JSONs, we remove the folder after a successful run to keep the tree clean.)
        try:
            shutil.rmtree(sub)
            print(f"Removed subfolder: {sub}")
        except OSError as e:
            print(f"Error removing subfolder {sub}: {e}")

        print(f"Subfolder '{sub.name}' done. JSONs moved: {moved_jsons}, files deleted: {deleted_files}")

    print("All subfolders processed.")


if __name__ == "__main__":
    main()
