# macros for mkdocs-macros-plugin
import importlib
import os
from datetime import datetime
from functools import cache
from typing import Any

import requests

_inline_code_styles = {
    ".py": "python",
    ".sh": "bash",
    ".h": "cpp",
    ".cpp": "cpp",
    ".c": "c",
    ".rs": "rs",
    ".js": "js",
    ".md": None,
}

# this is the overall record id, not a specific version
_NEWORDER_ZENODO_ID = 4031821  # search using this (or DOI 10.5281/zenodo.4031821) just doesnt work


@cache
def get_zenodo_record() -> dict[str, Any]:
    try:
        response = requests.get(
            "https://zenodo.org/api/records",
            params={
                "q": "(virgesmith) AND (neworder)",  # this is the only query that seems to work
                "access_token": os.getenv("ZENODO_PAT"),
            },
        )
        response.raise_for_status()
        # with open("zenodo-result.json", "w") as fd:
        #     fd.write(response.text)
        return response.json()["hits"]["hits"][0]
    except Exception as e:
        return {f"{e.__class__.__name__}": f"{e} while retrieving zenodo record"}


def write_requirements() -> None:
    try:
        with open("docs/requirements.txt", "w") as fd:
            fd.write(
                f"""\
# DO NOT EDIT
# auto-generated @ {datetime.now()} by docs/macros.py::write_requirements()
# required by readthedocs.io
"""
            )
            fd.writelines(
                f"{dep}=={importlib.metadata.version(dep)}\n"
                for dep in [
                    "mkdocs",
                    "mkdocs-macros-plugin",
                    "mkdocs-material",
                    "mkdocs-material-extensions",
                    "mkdocs-video",
                    "requests",
                ]
            )
    # ignore any error, this should only run in a dev env anyway
    except:  # noqa: E722
        pass


def define_env(env):
    @env.macro
    def insert_zenodo_field(*keys: str) -> Any:
        result = get_zenodo_record()
        for key in keys:
            result = result[key]
        return result

    @env.macro
    def include_snippet(filename, tag=None, show_filename=True):
        """looks for code in <filename> between lines containing "!<tag>!" """
        full_filename = os.path.join(env.project_dir, filename)

        _, file_type = os.path.splitext(filename)
        # default to literal "text" for inline code style
        code_style = _inline_code_styles.get(file_type, "text")

        with open(full_filename, "r") as f:
            lines = f.readlines()

        if tag:
            tag = f"!{tag}!"
            span = []
            for i, line in enumerate(lines):
                if tag in line:
                    span.append(i)
            if len(span) != 2:
                return f"```ERROR {filename} ({code_style}) too few/many tags ({len(span)}) for '{tag}'```"
            lines = lines[span[0] + 1 : span[1]]

        if show_filename:
            title = f'title="{filename}"'
        else:
            title = ""
        if code_style is not None:
            return f"```{code_style} {title}\n{''.join(lines)}```"
        else:
            return "".join(lines)


# write_requirements()
