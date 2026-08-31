import argparse
import os
import shutil
from pathlib import Path

_SKILL_NAME = "neworder"
_DEFAULT_PATH = ".agents"


def _source_dir() -> Path:
    """The bundled skill directory, shipped inside this package."""
    return Path(__file__).parent / "skill"


def _target_path(root: str) -> Path:
    return Path(root) / "skills" / _SKILL_NAME


def _is_ours(target: Path, source: Path) -> bool:
    """Whether target is a symlink to, or a copy of, the bundled skill - and so safe to replace or remove."""
    try:
        if target.is_symlink():
            return target.resolve() == source.resolve()
        if target.is_dir():
            shipped = {p.name for p in source.iterdir()}
            entries = list(target.iterdir())
            return bool(entries) and all(p.is_file() and p.name in shipped for p in entries)
    except OSError:
        return False
    return False


def _link_or_copy(source: Path, target: Path) -> str:
    # symlinks need developer mode or elevation on Windows, so fall back to copying there
    try:
        target.symlink_to(os.path.relpath(source.resolve(), target.parent.resolve()), target_is_directory=True)
    except OSError:
        shutil.copytree(source, target)
        return "copied"
    return "linked"


def _install(root: str) -> int:
    source = _source_dir()
    target = _target_path(root)
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.is_symlink() or target.exists():
        if not _is_ours(target, source):
            print(f"refusing to overwrite existing file or directory not managed by {_SKILL_NAME}-skill: {target}")
            return 1
        if target.is_symlink():
            print(f"already installed and up to date: {target}")
            return 0
        # a copy can go stale when the installed package is upgraded, so always refresh it
        shutil.rmtree(target)
        print(f"refreshed ({_link_or_copy(source, target)}): {target}")
        return 0

    print(f"installed ({_link_or_copy(source, target)}): {target} -> {source}")
    return 0


def _remove(root: str) -> int:
    source = _source_dir()
    target = _target_path(root)

    if not target.exists() and not target.is_symlink():
        print(f"not installed: {target}")
        return 0
    if not _is_ours(target, source):
        print(f"refusing to remove {target}: not managed by {_SKILL_NAME}-skill")
        return 1

    if target.is_symlink():
        target.unlink()
    else:
        shutil.rmtree(target)
    print(f"removed: {target}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the `neworder-skill` console script."""
    parser = argparse.ArgumentParser(
        prog=f"{_SKILL_NAME}-skill",
        description=f"Install or remove the '{_SKILL_NAME}' agent skill in a project.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--install",
        nargs="?",
        const=_DEFAULT_PATH,
        metavar="PATH",
        help=f"Install the skill into PATH/skills/{_SKILL_NAME} (default PATH: {_DEFAULT_PATH}).",
    )
    group.add_argument(
        "--remove",
        nargs="?",
        const=_DEFAULT_PATH,
        metavar="PATH",
        help=f"Remove the skill from PATH/skills/{_SKILL_NAME} (default PATH: {_DEFAULT_PATH}).",
    )
    args = parser.parse_args(argv)

    if args.install is not None:
        return _install(args.install)
    return _remove(args.remove)


if __name__ == "__main__":
    raise SystemExit(main())
