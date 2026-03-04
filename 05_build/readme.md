# 05_build

Build reusable retrieval corpora from vectorizers/models.

## MobileViT-XXS precompute

```bash
python 05_build/01_precompute_mobilevit_vectors.py
```

Defaults come from `.env` (`CCG_DATA_DIR` / `CCG_DATA_ROOT`) and store vectors under:

`<CCG_DATA_DIR>/vectors/mobilevit_xxs/img<image_size>/manifest_<manifest_name>/dataset_<dataset_name>/`

Example files:

- `mobilevit_xxs_base_320d_gallery.npz`
- `mobilevit_xxs_base_320d_query_solring.npz`
- `mobilevit_xxs_ft_e5_128d_gallery.npz`
- `mobilevit_xxs_ft_e5_128d_query_solring.npz`

Use `--rebuild-cache` to force regeneration.
