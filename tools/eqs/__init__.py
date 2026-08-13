"""Event Quality Score (EQS), a learned upstream fidelity axis.

EQS (Chanda et al., CVPRW'25, arXiv:2504.12515) feeds two event streams through a
pretrained RVT detection backbone and compares the first three stages'
activations in latent space.

The official repository releases only the vendored backbone, so ``eqs_score``
reimplements the metric from the paper's equations; the backbone, the
StackedHistogram representation, and the Gen1-small weights come from the
official RVT code.
"""
