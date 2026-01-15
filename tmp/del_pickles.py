import os

ROOT_DIR = "results/chkpt_runs/"


def get_directory_size(path):
    """Return total size (in bytes) of all files under path."""
    total_size = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total_size += os.path.getsize(fp)
            except OSError:
                pass
    return total_size


def human_readable(num_bytes):
    """Convert bytes to human-readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} PB"


def delete_pkl_files(path):
    """Delete all .pkl files under path."""
    deleted_files = 0
    deleted_bytes = 0

    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(".pkl"):
                fp = os.path.join(root, f)
                try:
                    size = os.path.getsize(fp)
                    os.remove(fp)
                    deleted_files += 1
                    deleted_bytes += size
                except OSError as e:
                    print(f"Could not delete {fp}: {e}")

    return deleted_files, deleted_bytes


if __name__ == "__main__":
    print(f"Scanning directory: {ROOT_DIR}")

    size_before = get_directory_size(ROOT_DIR)
    print(f"Storage before cleanup: {human_readable(size_before)}")

    deleted_files, deleted_bytes = delete_pkl_files(ROOT_DIR)

    size_after = get_directory_size(ROOT_DIR)
    print(f"Storage after cleanup:  {human_readable(size_after)}")

    print("\nSummary:")
    print(f"  Deleted files: {deleted_files}")
    print(f"  Freed space:   {human_readable(deleted_bytes)}")
