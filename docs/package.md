# Package

## Pip

Bump [VERSION](./VERSION), then build and test. If ok, run this script, which packages a source distribution and uploads it to PyPI (credentials required):

```bash
scripts/package.sh
```

Then, create a release on github, which will trigger zenodo generate a new DOI.

## Conda

Conda-force should automatically pick up any new release to pip. The feedstock is [here](https://github.com/conda-forge/neworder-feedstock)

## Docker

Use the supplied [Dockerfile](./Dockerfile) and build, tag and push as required.

## Documentation

`docs/macros.py` defines macros used by mkdocs to insert code (and other) snippets into files

readthedocs.io should auotmatically pick up and update the documentation on a commit. API documentation requires a manual step however:

```
scripts/docstr2md.py
```

regenerates markdown from the raw docstrings and writes to docs/apidoc.md, which is subsequently inserted into docs/api.md.