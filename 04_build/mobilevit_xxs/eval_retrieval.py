from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from data import ManifestRow, extract_card_id_from_filename


def default_eval_transform(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def embed_image_paths(model: torch.nn.Module, image_paths: list[str], device: torch.device, batch_size: int, image_size: int = 224) -> np.ndarray:
    tfm = default_eval_transform(image_size=image_size)
    vectors: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc="embed", unit="batch"):
            batch_paths = image_paths[i : i + batch_size]
            ims = []
            for p in batch_paths:
                im = Image.open(p).convert("RGB")
                ims.append(tfm(im))
            x = torch.stack(ims).to(device)
            z = model(x).cpu().numpy()
            vectors.append(z)
    if not vectors:
        return np.zeros((0, 1), dtype=np.float32)
    return np.concatenate(vectors, axis=0).astype(np.float32)


def recall_at_k(top_ids: list[list[str]], true_ids: list[str], k: int) -> float:
    if not true_ids:
        return 0.0
    hit = 0
    for pred, true_id in zip(top_ids, true_ids):
        if true_id in pred[:k]:
            hit += 1
    return hit / len(true_ids)


def eval_solring_retrieval(
    *,
    model: torch.nn.Module,
    manifest_rows: list[ManifestRow],
    solring_aligned_dir: Path,
    out_dir: Path,
    device: torch.device,
    batch_size: int = 32,
    image_size: int = 224,
) -> dict:
    gallery_rows = manifest_rows
    gallery_paths = [r.image_path for r in gallery_rows]
    gallery_ids = [r.card_id for r in gallery_rows]

    query_paths = sorted([str(p) for p in solring_aligned_dir.glob("*.jpg")])
    true_ids: list[str] = []
    query_paths_filtered: list[str] = []
    for qp in query_paths:
        cid = extract_card_id_from_filename(Path(qp).name)
        if cid:
            query_paths_filtered.append(qp)
            true_ids.append(cid)

    gallery_emb = embed_image_paths(model, gallery_paths, device=device, batch_size=batch_size, image_size=image_size)
    query_emb = embed_image_paths(model, query_paths_filtered, device=device, batch_size=batch_size, image_size=image_size)

    top_ids: list[list[str]] = []
    rows = []

    if len(true_ids) > 0 and query_emb.shape[0] > 0 and gallery_emb.shape[0] > 0:
        sims = query_emb @ gallery_emb.T
        order = np.argsort(-sims, axis=1)

        for i, qp in enumerate(query_paths_filtered):
            idxs = order[i, :10].tolist()
            preds = [gallery_ids[j] for j in idxs]
            top_ids.append(preds)
            rows.append(
                {
                    "query_image": qp,
                    "true_id": true_ids[i],
                    "top1_id": preds[0] if preds else "",
                    "top3_ids": preds[:3],
                    "top10_ids": preds[:10],
                }
            )

    metrics = {
        "top1": recall_at_k(top_ids, true_ids, 1),
        "top3": recall_at_k(top_ids, true_ids, 3),
        "top10": recall_at_k(top_ids, true_ids, 10),
        "n_queries": len(true_ids),
        "n_gallery": len(gallery_ids),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    if len(true_ids) == 0:
        metrics["warning"] = f"No Sol Ring queries found in {solring_aligned_dir}"

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "retrieval_summary.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with (out_dir / "retrieval_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["top1", "top3", "top10", "n_queries", "n_gallery", "evaluated_at", "warning"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow({k: metrics.get(k, "") for k in fields})

    with (out_dir / "retrieval_predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    return metrics
