---
language: 
- en
pretty_name: MoisesDB
tags:
- audio
- music
- source separation
license: other
license_name: cc-by-nc-sa-4.0
license_link: https://creativecommons.org/licenses/by-nc-sa/4.0/
---

# MoisesDB
Moises Dataset for Source Separation

### Dataset Description

- **Homepage:** [MoisesDB homepage](https://music.ai/research/)
- **Repository:** [MoisesDB repository](https://github.com/moises-ai/moises-db)
- **Paper:** [Moisesdb: A dataset for source separation beyond 4-stems](https://arxiv.org/abs/2307.15913)
- **Point of Contact:** [Igor Pereira](mailto:igor@moises.ai)

### Dataset Summary

MoisesDB is a dataset for source separation. It provides a collection of tracks and their separated stems (vocals, bass, drums, etc.). The dataset is used to evaluate the performance of source separation algorithms.

# Download the data

Please download the dataset at our research [website](https://music.ai/research/), extract it and configure the environment variable `MOISESDB_PATH` accordingly.

export MOISESDB_PATH=./moises-db-data

The directory structure should be

```
moisesdb:
    moisesdb_v0.1
        track uuid 0
        track uuid 1
        .
        .
        .
```

### Dataset Integrity

To verify the integrity of your downloaded dataset, you can check the following hashes:

#### MoisesDB
```
MD5: 13cf74eda129c38b914a51ea79fb1778
SHA256: 4cde33ce416ac7c868cffcb60eb31f5c741ab7ae5601cbb9d99ed498b72c48c1
```

#### Other dataset files available from Moises / Music.AI

Just for reference, not directly related to the tools in this repo.

SDXDB23_LabelNoise:
```
MD5: 629cfce51e4c8a36eae9c22aa5b710d3
SHA256: f6d2eac4ee1e21bf8237c0dcef2f3ebb9d04001ff8f999e7528107246eee08e2
```

SDXDB23_Bleeding:

```
MD5: be3ffafbdccb46b91507f73c44dabe4a
SHA256: b18a95da6b253bea986cf79990b6f2492d219871fdc17150ce599b45576d457e
```

#### How to Verify
You can verify the integrity on Linux/Mac using:
```bash
md5sum moisesdb.zip
sha256sum moisesdb.zip
```

Or on Windows using:
```powershell
Get-FileHash -Algorithm MD5 moisesdb.zip
Get-FileHash -Algorithm SHA256 moisesdb.zip
```

# Install

You can install this package with

```
pip install git+https://github.com/moises-ai/moises-db.git
```

# Usage

## `MoisesDB`

After downloading and configuring the path for the dataset, you can create an instance of `MoisesDB` to access the tracks. You can also provide the dataset path with the `data_path` argument.

```
from moisesdb.dataset import MoisesDB

db = MoisesDB(
    data_path='./moisesdb',
    sample_rate=44100
)
```

The `MoisesDB` object has iterator properties that you can use to access all files within the dataset.

```
n_songs = len(db)
track = db[0]  # Returns a MoisesDBTrack object
```

## `MoisesDBTrack`

The `MoisesDBTrack` object holds information about a track in the dataset, perform on-the-fly mixing for stems and multiple sources within a stem.

You can access all the stems and mixture from the `stem` and `audio` properties. The `stem` property returns a dictionary whith available stems as keys and `nd.array` on values. The `audio` property results in a `nd.array` with the mixture.

```
track = db[0]
stems = track.stems  # stems = {'vocals': ..., 'bass': ..., ...}
mixture track.audio # mixture = nd.array
```

The `MoisesDBTrack` object also contains other non-audio information from the track such as:
- `track.id`
- `track.provider`
- `track.artist`
- `track.name`
- `track.genre`
- `track.sources`
- `track.bleedings`
- `track.activity`

The stems and mixture are computed on-the-fly. You can create a stems-only version of the dataset using the `save_stems` method of the `MoisesDBTrack`.

```
track = db[0]
path =  './moises-db-stems/0'
track.save_stems(path)
```

# Performance Evaluation

We run a few source separation algorithms as well as oracle methods to evaluate the performance of each track of the `MoisesDB`. These results are located in `csv` files at the `benchmark` folder.

# Citing

If you used the `MoisesDB` dataset on your research, please cite the following paper.

```
@misc{pereira2023moisesdb,
      title={Moisesdb: A dataset for source separation beyond 4-stems}, 
      author={Igor Pereira and Felipe Araújo and Filip Korzeniowski and Richard Vogl},
      year={2023},
      eprint={2307.15913},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```

# Licensing

`MoisesDB` is distributed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0).

For the complete license terms, please visit: https://creativecommons.org/licenses/by-nc-sa/4.0/

See [LICENSE](LICENSE) file for details.