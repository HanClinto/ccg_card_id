from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import imagehash
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

_ROOT_RETRIEVAL = Path(__file__).resolve().parents[2]
if str(_ROOT_RETRIEVAL) not in sys.path:
    sys.path.insert(0, str(_ROOT_RETRIEVAL))

UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)


def extract_card_id(name: str) -> str | None:
    m = UUID_RE.search(name)
    return m.group(0).lower() if m else None


def eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class BackboneFeatureModel(torch.nn.Module):
    def __init__(self, backbone_name: str = "mobilevit_xxs"):
        super().__init__()
        import timm

        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0, global_pool="avg")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x)
        return torch.nn.functional.normalize(f, dim=1)


class FineTunedEmbeddingModel(torch.nn.Module):
    def __init__(self, backbone_name: str, embedding_dim: int):
        super().__init__()
        import timm

        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0, global_pool="avg")
        self.proj = torch.nn.Linear(self.backbone.num_features, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x)
        z = self.proj(f)
        return torch.nn.functional.normalize(z, dim=1)


def load_finetuned_model(checkpoint: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cargs = ckpt.get("args", {})
    # Support both single-task ("embedding_dim") and multitask ("embedding_dims") checkpoints.
    # Multitask shared-head checkpoints have the same backbone+proj structure; use first dim.
    emb_dim = cargs.get("embedding_dim") or (cargs.get("embedding_dims") or [128])[0]
    model = FineTunedEmbeddingModel(cargs.get("backbone", "mobilevit_xxs"), int(emb_dim))
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    return model, ckpt


def _cache_fingerprint(paths: list[Path], image_size: int) -> str:
    h = hashlib.sha1()
    h.update(str(image_size).encode("utf-8"))
    h.update(str(len(paths)).encode("utf-8"))
    for p in paths:
        h.update(str(p).encode("utf-8"))
    return h.hexdigest()


def embed_paths(
    model: torch.nn.Module,
    paths: list[Path],
    device: torch.device,
    batch_size: int,
    image_size: int,
    desc: str = "embed",
    cache_path: Path | None = None,
    rebuild_cache: bool = False,
) -> torch.Tensor:
    if cache_path is not None and cache_path.exists() and not rebuild_cache:
        try:
            data = np.load(cache_path, allow_pickle=False)
            emb = data["embeddings"].astype(np.float32)
            fp = str(data["fingerprint"]) if "fingerprint" in data.files else ""
            if fp == _cache_fingerprint(paths, image_size):
                print(f"{desc}: cache hit -> {cache_path}")
                return torch.from_numpy(emb)
            else:
                print(f"{desc}: cache stale (fingerprint mismatch) -> recomputing")
        except Exception:
            print(f"{desc}: cache unreadable -> recomputing")

    if cache_path is not None:
        if rebuild_cache:
            print(f"{desc}: rebuild requested -> recomputing embeddings")
        else:
            print(f"{desc}: cache miss -> computing embeddings")

    tfm = eval_transform(image_size)
    out: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(paths), batch_size), desc=desc, unit="batch"):
            batch = paths[i : i + batch_size]
            ims = [tfm(Image.open(p).convert("RGB")) for p in batch]
            x = torch.stack(ims).to(device)
            z = model(x).detach().to("cpu", dtype=torch.float32)
            out.append(z)
    if not out:
        emb_t = torch.zeros((0, 1), dtype=torch.float32)
    else:
        emb_t = torch.cat(out, dim=0)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, embeddings=emb_t.numpy(), fingerprint=_cache_fingerprint(paths, image_size))
        print(f"{desc}: saved cache {cache_path}")

    return emb_t


def _compute_hits(
    top_idxs_np,
    top_scores_np,
    gallery_ids: list[str],
    query_ids: list[str],
    query_paths: list[Path],
) -> tuple[dict, list[dict]]:
    """Compute top-1/3/10 hit rates and failures for one set of IDs."""
    hit1 = hit3 = hit10 = 0
    failures: list[dict] = []
    for i, true_id in enumerate(query_ids):
        idxs = top_idxs_np[i].tolist()
        top_ids = [gallery_ids[j] for j in idxs]
        top_scores = [float(x) for x in top_scores_np[i].tolist()]

        if true_id in top_ids[:1]:
            hit1 += 1
        if true_id in top_ids[:3]:
            hit3 += 1
        if true_id in top_ids[:10]:
            hit10 += 1

        if true_id not in top_ids[:1]:
            true_rank = next((r for r, cid in enumerate(top_ids, start=1) if cid == true_id), None)
            failures.append({
                "image_path": str(query_paths[i]),
                "true_id": true_id,
                "predicted_id": top_ids[0] if top_ids else "",
                "score": top_scores[0] if top_scores else None,
                "score_type": "cosine_similarity",
                "true_rank": true_rank,
            })

    n = len(query_ids)
    metrics = {"top1": hit1 / n, "top3": hit3 / n, "top10": hit10 / n,
               "n_queries": n, "n_gallery": len(gallery_ids)}
    return metrics, failures


def evaluate_retrieval(
    *,
    model: torch.nn.Module,
    gallery_paths: list[Path],
    query_paths: list[Path],
    # criteria: dict of criterion_name -> (gallery_ids, query_ids)
    # Supports any number of match criteria evaluated from a single embedding pass.
    # Standard keys: "edition" (card_id match), "artwork" (illustration_id match)
    criteria: dict[str, tuple[list[str], list[str]]],
    device: torch.device,
    batch_size: int,
    image_size: int,
    label: str = "model",
    gallery_label: str | None = None,
    gallery_cache_root: Path | None = None,
    query_cache_root: Path | None = None,
    rebuild_cache: bool = False,
    use_fp16: bool = True,
) -> dict[str, tuple[dict, list[dict]]]:
    """Embed gallery and queries once; evaluate all criteria from the same embeddings.

    gallery_label: cache key for gallery embeddings (defaults to label). Use the
    model variant name here so the gallery cache is shared across query datasets.

    Returns: dict of criterion_name -> (metrics_dict, failures_list)
    """
    _gallery_label = gallery_label if gallery_label is not None else label
    gallery_cache = gallery_cache_root / f"{_gallery_label}_gallery.npz" if gallery_cache_root is not None else None
    query_cache = query_cache_root / f"{label}_query.npz" if query_cache_root is not None else None

    g = embed_paths(
        model, gallery_paths, device=device, batch_size=batch_size,
        image_size=image_size, desc=f"{label}: gallery",
        cache_path=gallery_cache, rebuild_cache=rebuild_cache,
    )
    q = embed_paths(
        model, query_paths, device=device, batch_size=batch_size,
        image_size=image_size, desc=f"{label}: query",
        cache_path=query_cache, rebuild_cache=rebuild_cache,
    )

    # Empty result for all criteria if no data
    first_gallery_ids = next(iter(criteria.values()))[0] if criteria else []
    first_query_ids = next(iter(criteria.values()))[1] if criteria else []
    if not criteria or len(first_query_ids) == 0 or g.shape[0] == 0 or q.shape[0] == 0:
        return {
            name: ({"top1": 0.0, "top3": 0.0, "top10": 0.0,
                    "n_queries": len(q_ids), "n_gallery": len(g_ids)}, [])
            for name, (g_ids, q_ids) in criteria.items()
        }

    dtype = torch.float16 if (use_fp16 and device.type in {"mps", "cuda"}) else torch.float32
    g_dev = g.to(device=device, dtype=dtype)
    q_dev = q.to(device=device, dtype=dtype)

    sims = q_dev @ g_dev.T
    k = min(10, g_dev.shape[0])
    top_scores_t, top_idxs_t = torch.topk(sims, k=k, dim=1)
    top_scores_np = top_scores_t.to("cpu", dtype=torch.float32).numpy()
    top_idxs_np = top_idxs_t.to("cpu").numpy()

    results: dict[str, tuple[dict, list[dict]]] = {}
    for name, (gallery_ids, query_ids) in criteria.items():
        metrics, failures = _compute_hits(top_idxs_np, top_scores_np, gallery_ids, query_ids, query_paths)
        results[name] = (metrics, failures)
    return results


def load_manifest_gallery(manifest_csv: Path) -> tuple[list[Path], list[str], list[str]]:
    """Load gallery from manifest CSV.

    Returns: (paths, card_ids, illustration_ids)
    """
    import csv

    paths: list[Path] = []
    card_ids: list[str] = []
    illustration_ids: list[str] = []
    with manifest_csv.open("r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = Path(r["image_path"])
            if p.exists():
                paths.append(p)
                card_ids.append(str(r["card_id"]).lower())
                illustration_ids.append(str(r.get("illustration_id", "")).lower())
    return paths, card_ids, illustration_ids


def load_query_manifest(manifest_csv: Path) -> tuple[list[Path], list[str], list[str]]:
    """Load a query manifest CSV.

    Expected columns: image_path, card_id, illustration_id
    Missing images are silently skipped.

    Returns: (paths, card_ids, illustration_ids)
    """
    import csv

    paths: list[Path] = []
    card_ids: list[str] = []
    illustration_ids: list[str] = []
    with manifest_csv.open("r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = Path(r["image_path"])
            if not p.exists():
                continue
            paths.append(p)
            card_ids.append(str(r.get("card_id", "")).lower())
            illustration_ids.append(str(r.get("illustration_id", "")).lower())
    return paths, card_ids, illustration_ids


def load_solring_queries(dataset_dir: Path) -> tuple[list[Path], list[str]]:
    aligned = dataset_dir / "04_data" / "aligned"
    paths = sorted([p for p in aligned.glob("*.jpg")])
    q_paths: list[Path] = []
    q_ids: list[str] = []
    for p in paths:
        cid = extract_card_id(p.name)
        if cid:
            q_paths.append(p)
            q_ids.append(cid)
    return q_paths, q_ids


# ---------------------------------------------------------------------------
# pHash retrieval
# ---------------------------------------------------------------------------

def _phash_array(paths: list[Path], hash_size: int, desc: str) -> np.ndarray:
    """Compute phash for a list of images. Returns (n, bytes_per_hash) uint8."""
    bits = hash_size * hash_size
    bytes_per = (bits + 7) // 8
    out = np.zeros((len(paths), bytes_per), dtype=np.uint8)
    for i, p in enumerate(tqdm(paths, desc=desc, unit="img")):
        try:
            h = imagehash.phash(Image.open(p).convert("RGB"), hash_size=hash_size)
            out[i] = np.packbits(h.hash.flatten().astype(np.uint8))
        except Exception:
            pass  # leaves row as zeros
    return out


def compute_phash_embeddings(
    paths: list[Path],
    hash_size: int,
    desc: str = "phash",
    cache_path: Path | None = None,
    rebuild_cache: bool = False,
) -> np.ndarray:
    """Compute or load cached phash vectors. Returns (n, bytes) uint8 array."""
    fp = _cache_fingerprint(paths, hash_size)
    if cache_path is not None and cache_path.exists() and not rebuild_cache:
        try:
            data = np.load(cache_path, allow_pickle=False)
            if str(data.get("fingerprint", "")) == fp:
                print(f"{desc}: cache hit -> {cache_path}")
                return data["embeddings"]
            print(f"{desc}: cache stale -> recomputing")
        except Exception:
            print(f"{desc}: cache unreadable -> recomputing")
    else:
        if cache_path is not None:
            print(f"{desc}: {'rebuild requested' if rebuild_cache else 'cache miss'} -> computing")

    arr = _phash_array(paths, hash_size, desc)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, embeddings=arr, fingerprint=fp)
        print(f"{desc}: saved cache {cache_path}")
    return arr


def evaluate_phash_retrieval(
    *,
    gallery_paths: list[Path],
    query_paths: list[Path],
    criteria: dict[str, tuple[list[str], list[str]]],
    hash_size: int,
    gallery_label: str | None = None,
    label: str = "phash",
    gallery_cache_root: Path | None = None,
    query_cache_root: Path | None = None,
    rebuild_cache: bool = False,
) -> dict[str, tuple[dict, list[dict]]]:
    """Evaluate phash retrieval for multiple criteria from a single hash pass.

    Mirrors evaluate_retrieval() but uses Hamming distance on pHash vectors.
    """
    _gallery_label = gallery_label if gallery_label is not None else label
    gallery_cache = (
        gallery_cache_root / f"{_gallery_label}_gallery.npz"
        if gallery_cache_root is not None else None
    )
    query_cache = (
        query_cache_root / f"{label}_query.npz"
        if query_cache_root is not None else None
    )

    g = compute_phash_embeddings(
        gallery_paths, hash_size,
        desc=f"{label}: gallery",
        cache_path=gallery_cache, rebuild_cache=rebuild_cache,
    )
    q = compute_phash_embeddings(
        query_paths, hash_size,
        desc=f"{label}: query",
        cache_path=query_cache, rebuild_cache=rebuild_cache,
    )

    first_gallery_ids = next(iter(criteria.values()))[0] if criteria else []
    first_query_ids = next(iter(criteria.values()))[1] if criteria else []
    if not criteria or len(first_query_ids) == 0 or g.shape[0] == 0 or q.shape[0] == 0:
        return {
            name: ({"top1": 0.0, "top3": 0.0, "top10": 0.0,
                    "n_queries": len(q_ids), "n_gallery": len(g_ids)}, [])
            for name, (g_ids, q_ids) in criteria.items()
        }

    # Compute hamming distances once and reuse across all criteria.
    # gallery_paths is the same for all criteria; only the ID labels differ.
    from ccg_card_id.search.brute_force import _POPCOUNT_U8  # noqa: PLC0415
    n_q, n_g = len(q), len(g)
    k = min(10, n_g)
    distances = np.empty((n_q, n_g), dtype=np.int32)
    for i in range(n_q):
        xor = np.bitwise_xor(g, q[i])
        distances[i] = _POPCOUNT_U8[xor].sum(axis=1, dtype=np.int32)

    top_idxs_np = np.argsort(distances, axis=1)[:, :k]
    top_scores_np = distances[np.arange(n_q)[:, None], top_idxs_np].astype(np.float32)

    results: dict[str, tuple[dict, list[dict]]] = {}
    for name, (gallery_ids, query_ids) in criteria.items():
        metrics, failures = _compute_hits(
            top_idxs_np, top_scores_np, gallery_ids, query_ids, query_paths
        )
        results[name] = (metrics, failures)

    return results
