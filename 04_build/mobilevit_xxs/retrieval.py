from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

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
    model = FineTunedEmbeddingModel(cargs.get("backbone", "mobilevit_xxs"), int(cargs.get("embedding_dim", 128)))
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
                print(f"{desc}: loaded cache {cache_path}")
                return torch.from_numpy(emb)
        except Exception:
            pass

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


def evaluate_retrieval(
    *,
    model: torch.nn.Module,
    gallery_paths: list[Path],
    gallery_ids: list[str],
    query_paths: list[Path],
    query_ids: list[str],
    device: torch.device,
    batch_size: int,
    image_size: int,
    label: str = "model",
    cache_root: Path | None = None,
    rebuild_cache: bool = False,
    use_fp16: bool = True,
) -> tuple[dict, list[dict]]:
    gallery_cache = cache_root / f"{label}_gallery.npz" if cache_root is not None else None
    query_cache = cache_root / f"{label}_query.npz" if cache_root is not None else None

    g = embed_paths(
        model,
        gallery_paths,
        device=device,
        batch_size=batch_size,
        image_size=image_size,
        desc=f"{label}: gallery",
        cache_path=gallery_cache,
        rebuild_cache=rebuild_cache,
    )
    q = embed_paths(
        model,
        query_paths,
        device=device,
        batch_size=batch_size,
        image_size=image_size,
        desc=f"{label}: query",
        cache_path=query_cache,
        rebuild_cache=rebuild_cache,
    )

    failures: list[dict] = []
    if len(query_ids) == 0 or g.shape[0] == 0 or q.shape[0] == 0:
        return {"top1": 0.0, "top3": 0.0, "top10": 0.0, "n_queries": len(query_ids), "n_gallery": len(gallery_ids)}, failures

    dtype = torch.float16 if (use_fp16 and device.type in {"mps", "cuda"}) else torch.float32
    g_dev = g.to(device=device, dtype=dtype)
    q_dev = q.to(device=device, dtype=dtype)

    sims = q_dev @ g_dev.T
    k = min(10, g_dev.shape[0])
    top_scores_t, top_idxs_t = torch.topk(sims, k=k, dim=1)

    top_scores_np = top_scores_t.to("cpu", dtype=torch.float32).numpy()
    top_idxs_np = top_idxs_t.to("cpu").numpy()

    hit1 = hit3 = hit10 = 0
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
            failures.append(
                {
                    "image_path": str(query_paths[i]),
                    "true_id": true_id,
                    "predicted_id": top_ids[0] if top_ids else "",
                    "score": top_scores[0] if top_scores else None,
                    "score_type": "cosine_similarity",
                    "true_rank": true_rank,
                }
            )

    n = len(query_ids)
    metrics = {
        "top1": hit1 / n,
        "top3": hit3 / n,
        "top10": hit10 / n,
        "n_queries": n,
        "n_gallery": len(gallery_ids),
    }
    return metrics, failures


def load_manifest_gallery(manifest_csv: Path) -> tuple[list[Path], list[str]]:
    import csv

    paths: list[Path] = []
    ids: list[str] = []
    with manifest_csv.open("r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = Path(r["image_path"])
            if p.exists():
                paths.append(p)
                ids.append(str(r["card_id"]).lower())
    return paths, ids


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
