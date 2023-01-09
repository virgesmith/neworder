# Citing *neworder*

If you use `neworder` in any published work, please cite it. You can use either of the following Bibtex references:

## The [JOSS paper](https://joss.theoj.org/papers/10.21105/joss.03351)

```bibtex
@article{Smith2021,
  doi = {10.21105/joss.03351},
  url = {https://doi.org/10.21105/joss.03351},
  year = {2021},
  publisher = {The Open Journal},
  volume = {6},
  number = {63},
  pages = {3351},
  author = {Andrew P. Smith},
  title = {neworder: a dynamic microsimulation framework for Python},
  journal = {Journal of Open Source Software}
}
```

## The package itself

```bibtex
@software{neworder,
   doi = { {{ insert_zenodo_field("doi") }} },
   author = { Andrew P Smith },
   year = { {{ insert_zenodo_field("metadata", "publication_date")[:4] }} },
   version = { {{ insert_zenodo_field("metadata", "version") }} },
   url = { https://neworder.readthedocs.io/en/stable/ },
   title = { neworder: a dynamic microsimulation framework for Python }
}
```
