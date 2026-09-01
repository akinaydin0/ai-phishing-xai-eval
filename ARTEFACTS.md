# Heavy artefacts of this repository

Everything that is too large for plain git (GitHub rejects files over 100 MB) is stored as
assets of the GitHub Release `artefacts-v1`, created by `tools/archive_artefacts.ps1` and
restored into place by `tools/restore_artefacts.ps1`. Every archive was zipped from the
repository root, so extracting it in a fresh clone recreates the original paths. A SHA-256
manifest (`MANIFEST.sha256`) is uploaded alongside and checked on restore.

| Archive | Restores | Approx. size |
|---|---|---|
| `ckpt256__<model>__<corpus>.zip` (9) | `results/transformers/checkpoints/<model>__<corpus>/` | 255 to 476 MB each |
| `ckpt512__<model>__<corpus>.zip` (6) | `results/transformers_512/checkpoints/<model>__<corpus>/` | 255 to 476 MB each |
| `data_cache.zip` | `data/cache/` (cleaned corpora, pinned universes) | 352 MB |
| `data_processed.zip` | `data/processed/` | 51 MB |
| `data_raw.zip` | `data/raw/` (D1 Kaggle CSVs, D2 Enron-Spam mail files, D3 PhishFuzzer JSON) | see manifest |
| `data_embeddings_glove.zip` | `data/embeddings/glove.6B.300d.txt` | 990 MB (also a public download) |
| `preproc_ablation_cache.zip` | `results/preproc_ablation/cache/` | 776 MB |

Everything else (source, frozen splits, all summary results, XAI attribution JSONs, statistics,
human-validation outputs, notes and the decision log, figures, the viva package) is committed
directly in git. The Python environment is not archived: recreate it from `requirements.txt`
(pinned: PyTorch 2.11.0+cu128, Transformers 4.57.6, Accelerate 1.14.0, scikit-learn 1.8.0).

Restore on a new machine:

    git clone https://github.com/akinaydin0/msc-thesis.git
    cd msc-thesis
    gh auth login          # once
    .\tools\restore_artefacts.ps1
