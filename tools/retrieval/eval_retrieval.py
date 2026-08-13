"""Cosine retrieval R@1/5/10 and per-query AP between two embedding sets.

Both inputs must hold L2-normalized embeddings so cosine similarity reduces to a
single matmul. Relevance is same-class; same-recording self-hits are masked, which
matters when query and gallery share a source.

Writes per-query arrays plus a sidecar JSON summary. The bootstrap CI always
re-aggregates from the per-query arrays, never from the sidecar means. Query order
is persisted because the pairing across seeds depends on it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

K_LIST = (1, 5, 10)


def load_embeddings(npz_path: Path) -> dict:
    z = np.load(npz_path)  # default allow_pickle=False is fine — pure ndarray
    out = {
        "prefix": np.asarray(z["prefix"]),
        "embedding": z["embedding"].astype(np.float32),
        "label": z["label"].astype(np.int64),
        "path": str(npz_path),
    }
    sidecar = Path(npz_path).with_suffix(".meta.json")
    out["meta"] = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    return out


def per_query_metrics(
    query_emb: np.ndarray,
    gallery_emb: np.ndarray,
    query_label: np.ndarray,
    gallery_label: np.ndarray,
    query_prefix: np.ndarray,
    gallery_prefix: np.ndarray,
    k_list: tuple[int, ...] = K_LIST,
) -> dict:
    """Return per-query R@K (binary) and AP (continuous).

    Algorithm:
      1. sim = Q @ G.T  (cosine sim because both are L2-normalized)
      2. Mask same-prefix self-hits with -inf so they cannot be top-K.
      3. argsort descending → rank matrix.
      4. relevance[q, j] = (label[q] == label[ranked_j]).
      5. R@K = any(relevance[:, :K], axis=1).
      6. AP_q = sum over each TP of (precision_at_that_rank) / total_relevant_q.
    """
    assert query_emb.shape[1] == gallery_emb.shape[1]
    Nq = query_emb.shape[0]
    Ng = gallery_emb.shape[0]

    sim = query_emb @ gallery_emb.T  # (Nq, Ng), float32

    # Mask same-prefix entries with -inf so they sort to the bottom.
    same_prefix = query_prefix[:, None] == gallery_prefix[None, :]
    if same_prefix.any():
        sim = np.where(same_prefix, -np.inf, sim)

    # Rank gallery by descending sim per query.
    order = np.argsort(-sim, axis=1)  # (Nq, Ng), int64
    ranked_labels = gallery_label[order]  # (Nq, Ng)
    relevance = (ranked_labels == query_label[:, None]).astype(np.float32)

    # R@K per query.
    r_at = np.zeros((Nq, len(k_list)), dtype=np.float32)
    for i, k in enumerate(k_list):
        r_at[:, i] = (relevance[:, :k].sum(axis=1) > 0).astype(np.float32)

    # Per-query AP.
    cumhits = np.cumsum(relevance, axis=1)
    ranks = np.arange(1, Ng + 1, dtype=np.float32)[None, :]
    precision_at = cumhits / ranks
    n_relevant = relevance.sum(axis=1)
    ap = np.where(
        n_relevant > 0,
        (precision_at * relevance).sum(axis=1) / np.maximum(n_relevant, 1.0),
        0.0,
    ).astype(np.float32)

    return {
        "r_at_k": r_at,
        "ap": ap,
        "n_query": Nq,
        "n_gallery": Ng,
        "n_self_masked": int(same_prefix.sum()),
    }


def main() -> int:
    ap_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_parser.add_argument("--query", type=Path, required=True)
    ap_parser.add_argument("--gallery", type=Path, required=True)
    ap_parser.add_argument("--output", type=Path, required=True, help="per-query metrics NPZ")
    args = ap_parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    Q = load_embeddings(args.query)
    G = load_embeddings(args.gallery)
    print(
        f"[eval] query  : {args.query}  shape={Q['embedding'].shape}  "
        f"({Q['meta'].get('variant')}, {Q['meta'].get('modality')})\n"
        f"[eval] gallery: {args.gallery}  shape={G['embedding'].shape}  "
        f"({G['meta'].get('variant')}, {G['meta'].get('modality')})",
        flush=True,
    )

    res = per_query_metrics(
        Q["embedding"], G["embedding"],
        Q["label"], G["label"],
        Q["prefix"], G["prefix"],
    )
    means = {
        "R@1": float(res["r_at_k"][:, 0].mean()),
        "R@5": float(res["r_at_k"][:, 1].mean()),
        "R@10": float(res["r_at_k"][:, 2].mean()),
        "mAP": float(res["ap"].mean()),
    }
    print(
        f"[eval] mean · R@1={means['R@1']*100:6.2f}  R@5={means['R@5']*100:6.2f}  "
        f"R@10={means['R@10']*100:6.2f}  mAP={means['mAP']*100:6.2f}  "
        f"(self-masked {res['n_self_masked']} cells)",
        flush=True,
    )

    np.savez_compressed(
        args.output,
        query_prefix=Q["prefix"],
        query_label=Q["label"].astype(np.int8),
        r_at_k=res["r_at_k"].astype(np.float32),
        ap=res["ap"].astype(np.float32),
    )
    sidecar = args.output.with_suffix(".meta.json")
    sidecar.write_text(json.dumps({
        "k_list": list(K_LIST),
        "query_npz": str(args.query),
        "gallery_npz": str(args.gallery),
        "query_meta": Q["meta"],
        "gallery_meta": G["meta"],
        "n_self_masked": res["n_self_masked"],
        "means": means,
    }, indent=2))
    print(f"[eval] wrote per-query metrics → {args.output}\n[eval] sidecar meta → {sidecar}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
