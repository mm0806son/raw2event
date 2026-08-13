"""v2e baseline comparison toolchain.

Generates events with v2e (Hu et al., CVPRW 2021) on the same Pi RAW / RGB
videos used by Raw2Event (DVS-Voltmeter), routes them through the existing
AprilTag-driven event_filter pipeline + unified80 rescale, and produces NPZ
datasets in the same format as Raw2Event's `*_filtered_*.npz`.

The locked 12-variant reference matrix (V01-V12) is described in
the twelve-variant reference simulator matrix.
"""
