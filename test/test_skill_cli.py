from pathlib import Path

import pytest

from neworder import skill_cli


def _symlinks_available(tmp_path: Path) -> bool:
    probe = tmp_path / "probe"
    try:
        probe.symlink_to(tmp_path, target_is_directory=True)
    except OSError:
        return False
    probe.unlink()
    return True


def _no_symlinks(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_oserror(*_args: object, **_kwargs: object) -> None:
        raise OSError("symlinks not permitted")

    monkeypatch.setattr(Path, "symlink_to", raise_oserror)


def test_install_creates_skill(tmp_path: Path) -> None:
    assert skill_cli.main(["--install", str(tmp_path)]) == 0

    target = tmp_path / "skills" / "neworder"
    assert (target / "SKILL.md").is_file()
    if _symlinks_available(tmp_path):
        assert target.is_symlink()
        assert target.resolve() == skill_cli._source_dir().resolve()


def test_install_default_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    assert skill_cli.main(["--install"]) == 0

    assert (tmp_path / ".agents" / "skills" / "neworder" / "SKILL.md").is_file()


def test_install_idempotent(tmp_path: Path) -> None:
    assert skill_cli.main(["--install", str(tmp_path)]) == 0
    assert skill_cli.main(["--install", str(tmp_path)]) == 0

    assert (tmp_path / "skills" / "neworder" / "SKILL.md").is_file()


def test_install_refuses_existing_directory(tmp_path: Path) -> None:
    target = tmp_path / "skills" / "neworder"
    target.mkdir(parents=True)
    (target / "keepme.txt").write_text("do not delete")

    assert skill_cli.main(["--install", str(tmp_path)]) == 1
    assert not target.is_symlink()
    assert (target / "keepme.txt").read_text() == "do not delete"


def test_install_refuses_foreign_symlink(tmp_path: Path) -> None:
    if not _symlinks_available(tmp_path):
        pytest.skip("symlinks not available on this platform")
    foreign = tmp_path / "elsewhere"
    foreign.mkdir()
    target = tmp_path / "skills" / "neworder"
    target.parent.mkdir(parents=True)
    target.symlink_to(foreign, target_is_directory=True)

    assert skill_cli.main(["--install", str(tmp_path)]) == 1
    assert target.resolve() == foreign.resolve()


def test_install_copies_when_symlinks_unavailable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _no_symlinks(monkeypatch)
    assert skill_cli.main(["--install", str(tmp_path)]) == 0

    target = tmp_path / "skills" / "neworder"
    assert not target.is_symlink()
    assert (target / "SKILL.md").read_text() == (skill_cli._source_dir() / "SKILL.md").read_text()


def test_install_across_windows_drives(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # os.path.relpath raises when the package and the project are on different drives
    def raise_valueerror(*_args: object, **_kwargs: object) -> str:
        raise ValueError("path is on mount 'D:', start on mount 'C:'")

    monkeypatch.setattr(skill_cli.os.path, "relpath", raise_valueerror)
    assert skill_cli.main(["--install", str(tmp_path)]) == 0

    target = tmp_path / "skills" / "neworder"
    assert (target / "SKILL.md").read_text() == (skill_cli._source_dir() / "SKILL.md").read_text()
    assert skill_cli.main(["--remove", str(tmp_path)]) == 0
    assert not target.exists()


def test_install_refreshes_stale_copy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _no_symlinks(monkeypatch)
    skill_cli.main(["--install", str(tmp_path)])
    target = tmp_path / "skills" / "neworder"
    (target / "SKILL.md").write_text("stale content from an older version")

    assert skill_cli.main(["--install", str(tmp_path)]) == 0
    assert (target / "SKILL.md").read_text() == (skill_cli._source_dir() / "SKILL.md").read_text()


def test_remove_deletes_installed_skill(tmp_path: Path) -> None:
    skill_cli.main(["--install", str(tmp_path)])
    target = tmp_path / "skills" / "neworder"
    assert target.exists()

    assert skill_cli.main(["--remove", str(tmp_path)]) == 0
    assert not target.exists()
    assert not target.is_symlink()


def test_remove_deletes_copy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _no_symlinks(monkeypatch)
    skill_cli.main(["--install", str(tmp_path)])
    target = tmp_path / "skills" / "neworder"

    assert skill_cli.main(["--remove", str(tmp_path)]) == 0
    assert not target.exists()


def test_remove_missing_is_noop(tmp_path: Path) -> None:
    assert skill_cli.main(["--remove", str(tmp_path)]) == 0


def test_remove_refuses_foreign_directory(tmp_path: Path) -> None:
    target = tmp_path / "skills" / "neworder"
    target.mkdir(parents=True)
    (target / "keepme.txt").write_text("do not delete")

    assert skill_cli.main(["--remove", str(tmp_path)]) == 1
    assert (target / "keepme.txt").read_text() == "do not delete"


def test_remove_default_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    skill_cli.main(["--install"])
    target = tmp_path / ".agents" / "skills" / "neworder"
    assert target.exists()

    assert skill_cli.main(["--remove"]) == 0
    assert not target.exists()


def test_mutually_exclusive_args_required() -> None:
    with pytest.raises(SystemExit):
        skill_cli.main([])
    with pytest.raises(SystemExit):
        skill_cli.main(["--install", ".", "--remove", "."])
