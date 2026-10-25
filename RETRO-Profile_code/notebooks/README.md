# Notebooks

Place exploratory and visualization notebooks here. Suggested starters:

  * `01_explore_timit.ipynb` — sanity-check label distributions, durations,
    bag-size statistics.
  * `02_visualize_embeddings.ipynb` — UMAP / PCA of the projector output
    coloured by trait, before vs after OrdCon.
  * `03_neighbour_explanations.ipynb` — load a trained model + datastore and
    inspect the kNN explanation for individual test speakers.

The codebase is fully importable as a package, so any notebook in this
directory can do:

```python
from retro_profile.data import TimitDataModule
from retro_profile.models import RetroProfile
```

once you `pip install -e .` from the repo root.
