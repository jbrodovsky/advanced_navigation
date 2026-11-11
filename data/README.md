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
| `drone-data/` | 1: visual-inertial drone pipeline | ~545 MB | [course data bundle](https://drive.google.com/file/d/1Rx2NxDNcyMqUBWyMIPeASvCeCK0BIW_h/view?usp=drive_link) (`Project1-data/`) |
| `victoria-park/` | 2: Victoria Park FastSLAM | ~12 MB | [course data bundle](https://drive.google.com/file/d/1Rx2NxDNcyMqUBWyMIPeASvCeCK0BIW_h/view?usp=drive_link) (`Project2-data/`) |
| `mems-nav-dataset/` | 3: INS/GNSS integration | ~2.5 GB | [course data bundle](https://drive.google.com/file/d/1Rx2NxDNcyMqUBWyMIPeASvCeCK0BIW_h/view?usp=drive_link) (`Project3-data/`) |

All three datasets share one download, `adv_nav_data.zip` (about 1.6 GB);
`all` fetches it once. If the script cannot download it, get the zip in a
browser from the link above and install it with:

```bash
uv run python scripts/fetch_data.py all --archive ~/Downloads/adv_nav_data.zip
```

To keep the data elsewhere, set `ADVNAV_DATA_DIR` to a folder that contains
these dataset folders. The starter packages and the book look there first.

`.downloads/` holds a partly fetched archive, and `_upload/` holds archives built
by `scripts/package_datasets.py`. Both are ignored by git.

## Credits

- **Victoria Park:** J. Guivant and E. Nebot, Australian Centre for Field
  Robotics, University of Sydney.
- **MEMS navigation dataset:** J. Brodovsky (see the dataset repository).
- **Drone dataset:** provenance and terms to be confirmed (GitHub issue #3).

These datasets are provided for course use only. Do not redistribute them.
