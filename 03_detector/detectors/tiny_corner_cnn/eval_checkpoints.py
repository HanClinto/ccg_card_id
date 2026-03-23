#!/usr/bin/env python3
"""Evaluate saved checkpoints with CPE + pHash dewarp accuracy.

Useful for retroactively computing the pHash metric on checkpoints that were
saved before that metric was added to train.py.

Usage (run from project root):
    python 03_detector/detectors/tiny_corner_cnn/eval_checkpoints.py \
        results/corner_detector/mvit_spatialreg4_lc5_img448_ph10_seedin_blr10_fz2/epoch_0005.pt \
        results/corner_detector/mvit_spatialreg4_lc5_img448_ph10_seedin_blr10_fz2/epoch_0010.pt

    # Or glob a whole run directory
    python 03_detector/detectors/tiny_corner_cnn/eval_checkpoints.py \
        results/corner_detector/mvit_spatialreg4_lc5_img448_ph10_seedin_blr10_fz2/epoch_*.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(_HERE))

from ccg_card_id.config import cfg
from model import MobileViTCornerDetector, TinyCornerCNN, TinyCornerCNNDirect
from dataset import (
    CornerDataset,
    load_from_packopening_db,
    load_clint_as_test,
)
from train import (
    _val_cpe,
    _val_iou,
    _val_presence_acc,
    build_ref_phash_dict,
    batch_phash_dists,
    pick_device,
)


def eval_checkpoint(
    ckpt_path: Path,
    val_dl: DataLoader,
    test_dl: DataLoader | None,
    ref_phash_dict: dict,
    device: torch.device,
) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = ckpt.get("arch", "mobilevit")

    if arch == "tiny":
        model = TinyCornerCNN()
    elif arch == "tiny-direct":
        model = TinyCornerCNNDirect()
    else:
        pool_size = ckpt.get("pool_size", 4)
        model = MobileViTCornerDetector(pretrained_backbone=False, pool_size=pool_size)

    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    def _run_dl(dl):
        pc_all, tc_all, pp_all, tp_all = [], [], [], []
        ph_sum = ph_total = 0
        with torch.no_grad():
            for batch in tqdm(dl, desc=f"  eval {ckpt_path.name}", leave=False):
                images   = batch["image"].to(device)
                presence = batch["card_present"].to(device)
                corners  = batch["corners"].to(device)
                card_ids = batch.get("card_id", [])

                out = model(images)
                pc, pp = out[0], out[1]
                pc_all.append(pc.cpu()); tc_all.append(corners.cpu())
                pp_all.append(pp.cpu()); tp_all.append(presence.cpu())

                if ref_phash_dict and card_ids:
                    s, n = batch_phash_dists(
                        pc, images, card_ids, presence.bool(), ref_phash_dict,
                    )
                    ph_sum += s; ph_total += n

        pc_cat = torch.cat(pc_all); tc_cat = torch.cat(tc_all)
        pp_cat = torch.cat(pp_all); tp_cat = torch.cat(tp_all)
        return {
            "cpe":             _val_cpe(pc_cat, tc_cat, tp_cat.bool()),
            "iou":             _val_iou(pc_cat, tc_cat, tp_cat.bool()),
            "pres_acc":        _val_presence_acc(pp_cat, tp_cat),
            "mean_phash_dist": ph_sum / ph_total if ph_total > 0 else None,
            "ph_total":        ph_total,
        }

    val_metrics  = _run_dl(val_dl)
    test_metrics = _run_dl(test_dl) if test_dl is not None else {}

    return {"val": val_metrics, "test": test_metrics, "epoch": ckpt.get("epoch", "?")}


def main() -> None:
    p = argparse.ArgumentParser(description="Eval corner-detector checkpoints with CPE + pHash")
    p.add_argument("checkpoints", nargs="+", type=Path, help="Checkpoint .pt files to evaluate")
    p.add_argument("--data-dir", type=Path, default=cfg.data_dir)
    p.add_argument("--fast-data-dir", type=Path,
                   default=cfg.fast_data_dir if cfg.fast_data_dir.exists() else None)
    p.add_argument("--max-phash-dist", type=int, default=10,
                   help="Training data quality filter (default: 10)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    data_dir = args.data_dir
    db_path  = data_dir / "datasets/packopening/packopening.db"
    clint_csv = data_dir / "datasets/clint_cards_with_backgrounds/data/04_data/corners.csv"
    clint_neg = data_dir / "datasets/clint_cards_with_backgrounds/data/04_data/bad"

    print("Loading datasets...")
    _, val_rows = load_from_packopening_db(
        db_path, data_dir,
        neg_sample_n=0,
        val_frac=0.05,
        max_phash_dist=args.max_phash_dist,
    )
    test_rows = load_clint_as_test(clint_csv, clint_neg, data_dir) if clint_csv.exists() else []

    val_ds  = CornerDataset(val_rows,  data_dir, augment=False, fast_data_dir=args.fast_data_dir)
    test_ds = CornerDataset(test_rows, data_dir, augment=False) if test_rows else None

    dl_kwargs = dict(num_workers=args.num_workers, pin_memory=False,
                     persistent_workers=(args.num_workers > 0))
    val_dl  = DataLoader(val_ds,  batch_size=args.batch_size, shuffle=False, **dl_kwargs)
    test_dl = (DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **dl_kwargs)
               if test_ds else None)

    all_eval_ids = {r.get("card_id", "") for r in val_rows + test_rows if r.get("card_id")}
    print(f"Building ref pHash dict: {len(all_eval_ids)} unique cards...", end=" ", flush=True)
    ref_phash_dict = build_ref_phash_dict(all_eval_ids, data_dir, cache_dir=args.fast_data_dir)
    print(f"{len(ref_phash_dict)} found")

    device = pick_device(args.cpu)
    print(f"Device: {device}\n")

    # Sort checkpoints by epoch number if names contain digits
    ckpt_paths = sorted(args.checkpoints, key=lambda p: p.stem)

    print(f"{'Checkpoint':<55} {'Ep':>3}  {'val_cpe':>8}  {'val_iou':>7}  {'val_phash':>9}  {'test_cpe':>8}  {'test_iou':>7}  {'test_phash':>10}")
    print("-" * 125)

    for ckpt_path in ckpt_paths:
        if not ckpt_path.exists():
            # Try relative to data_dir results
            ckpt_path = data_dir / "results" / "corner_detector" / ckpt_path
        if not ckpt_path.exists():
            print(f"  NOT FOUND: {ckpt_path}")
            continue

        result = eval_checkpoint(
            ckpt_path, val_dl, test_dl, ref_phash_dict, device,
        )
        ep = result["epoch"]
        v  = result["val"]
        t  = result.get("test", {})

        val_ph_str   = f"{v['mean_phash_dist']:.1f}" if v.get("mean_phash_dist") is not None else "n/a"
        test_cpe_s   = f"{t['cpe']:.4f}" if t.get("cpe") is not None else "    n/a"
        test_iou_s   = f"{t['iou']:.3f}" if t.get("iou") is not None else "   n/a"
        test_ph_str  = f"{t['mean_phash_dist']:.1f}" if t.get("mean_phash_dist") is not None else "n/a"

        print(
            f"{ckpt_path.name:<55} {ep:>3}  {v['cpe']:>8.4f}  {v['iou']:>7.3f}  {val_ph_str:>9}  "
            f"{test_cpe_s:>8}  {test_iou_s:>7}  {test_ph_str:>10}"
        )


if __name__ == "__main__":
    main()
