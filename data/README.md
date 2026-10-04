# data/

Course datasets live here but are **not stored in git**. Download them from the
repository root:

```bash
uv run python scripts/fetch_data.py --list   # what exists and what is downloaded
uv run python scripts/fetch_data.py all      # everything
uv run python scripts/fetch_data.py drone-data
```

| Folder | Project | Size | Source |
|---|---|---|---|
| `drone-data/` | 1: visual-inertial drone pipeline | ~570 MB | course share link (`scripts/data_sources.toml`) |
| `victoria-park/` | 2: Victoria Park FastSLAM | ~1.5 MB | course share link (`scripts/data_sources.toml`) |
| `mems-nav-dataset/` | 3: INS/GNSS integration | ~150 MB | <https://github.com/jbrodovsky/mems-nav-dataset> |

If a download link has not been published yet, the script says so and points to
the GitHub issue tracking it. If you received an archive another way, install it
with:

```bash
uv run python scripts/fetch_data.py victoria-park --archive ~/Downloads/victoria-park.zip
```

To keep the data elsewhere, set `ADVNAV_DATA_DIR` to a folder that contains
these dataset folders. The starter packages and the book look there first.

`_upload/` holds the archives built by `scripts/package_datasets.py` for the
instructor to upload. It is also ignored by git.

## Credits

- **Victoria Park:** J. Guivant and E. Nebot, Australian Centre for Field
  Robotics, University of Sydney.
- **MEMS navigation dataset:** J. Brodovsky (see the dataset repository).
- **Drone dataset:** provenance and terms to be confirmed (GitHub issue #3).

These datasets are provided for course use only. Do not redistribute them.
