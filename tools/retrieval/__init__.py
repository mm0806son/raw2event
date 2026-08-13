"""Frozen-feature cross-modal event retrieval.

Tools in this package:
    extract_embeddings.py  — load a QKFormer ckpt, hook penultimate (B, 256)
                              feature, dump per-prefix L2-normalized NPZ.
    eval_retrieval.py      — cosine retrieval R@1/5/10 + per-query AP between
                              two embedding NPZs (query + gallery).
    compute_retrieval_ci.py— paired bootstrap CI over 559 prefixes, 8-row
                              wide markdown table + full JSON dump.

Statistical protocol mirrors tools/v2e_baseline/cross_modal_eval_with_ci.py:
B=1000 paired bootstrap on N=559 prefixes; per-seed-pair stack (S=3, N).
"""
