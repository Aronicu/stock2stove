import os
import shutil
import stat


def get_project_root():
    return os.path.dirname(os.path.abspath(__file__))


def handle_remove_readonly(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def remove_dir(path):
    if os.path.isdir(path):
        print(f"[CLEAN] Removing {path}")
        shutil.rmtree(path, onerror=handle_remove_readonly)


def remove_pycache(root_dir):
    for current_root, dirs, _ in os.walk(root_dir):
        if "__pycache__" in dirs:
            cache_path = os.path.join(current_root, "__pycache__")
            remove_dir(cache_path)
            dirs.remove("__pycache__")


def main():
    project_root = get_project_root()
    targets = [
        os.path.join(project_root, ".venv"),
        os.path.join(project_root, ".certs"),
        os.path.join(project_root, "frontend", "node_modules"),
        os.path.join(project_root, "frontend", "dist"),
    ]
    for target in targets:
        remove_dir(target)
    remove_pycache(project_root)
    print("[CLEAN] Done")


if __name__ == "__main__":
    main()
