"""Configuration for DatasetH5_all_queryctx.

Provides a self-contained argument parser for the queryctx dataset.
Supports both standalone usage and compatibility with external argparse scripts
via ``parse_known_args()``.

Usage:
    from queryctx_module.config.data_config import queryctx_args
    args = queryctx_args()           # standalone parse
    args, _ = queryctx_args()        # same (returns args only)
"""

import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}")


_parser = argparse.ArgumentParser(add_help=False)

_g = _parser.add_argument_group("H5 data files")
_g.add_argument(
    "--h5File",
    type=str,
    default="dongfang_field1031/raw5d_data1104.h5",
    help="Path to irregular (observed) H5 file",
)
_g.add_argument(
    "--h5File_regular",
    type=str,
    default="dongfang_field1031/reg5dbin_label1031.h5",
    help="Path to regular-grid H5 file (used for coord stats and inference query data)",
)

_g = _parser.add_argument_group("Data shape")
_g.add_argument("--time_ps", type=int, default=1256, help="Number of time samples per trace")
_g.add_argument("--trace_ps", type=int, default=128, help="Patch trace count (Q + K)")

_g = _parser.add_argument_group("Training / inference flags")
_g.add_argument("--train", type=str2bool, default=True, help="Training mode")
_g.add_argument("--expand", type=float, default=0.1)

_g = _parser.add_argument_group("Missing ratio (unused in queryctx, kept for compat)")
_g.add_argument("--min_r", type=float, default=0.4)
_g.add_argument("--max_r", type=float, default=0.7)

# ---- queryctx-specific ----
_g = _parser.add_argument_group("Query-Context dataset (DatasetH5_all_queryctx)")
_g.add_argument(
    "--dataset_neighbors_train",
    type=str,
    default=None,
    help="queryctx train npz path (pool_idx_2d.npz)",
)
_g.add_argument(
    "--dataset_neighbors_test",
    type=str,
    default=None,
    help="queryctx test npz path (infer_query_context.npz)",
)
_g.add_argument(
    "--train_num_query",
    type=int,
    default=16,
    help="Number of query traces per training sample",
)
_g.add_argument(
    "--train_context_size",
    type=int,
    default=None,
    help="Fixed context size (None = trace_ps - train_num_query)",
)
_g.add_argument(
    "--patch_beta",
    type=float,
    default=0.3,
    help="Diversity weight for context selection (diverse_topk beta)",
)
_g.add_argument(
    "--force_anchor_query",
    type=str2bool,
    default=False,
    help="Always include anchor in query set",
)
_g.add_argument(
    "--trace_sort_keys",
    type=str,
    default="offset,azimuth",
    help="Comma-separated coordinate sort order (offset,azimuth,sx,sy,rx,ry)",
)
_g.add_argument(
    "--epoch_repeat",
    type=int,
    default=4,
    help="Repeat each anchor this many times per epoch (with different random seeds for diversity)",
)
_g.add_argument(
    "--use_phys_omega",
    type=str2bool,
    default=False,
    help="Use physics-based omega to set RoPE base frequency",
)
_g.add_argument("--use_p_scale", type=str2bool, default=True, help="Scale coordinates by p_scale")

# ---- dataset_type (for switching) ----
_g.add_argument(
    "--dataset_type",
    type=str,
    default="queryctx",
    choices=["interp", "interp_4d", "queryctx"],
    help="Dataset type selector",
)


def get_parser():
    """Return the raw ArgumentParser instance (for use as parent parser)."""
    return _parser


def queryctx_args(args_list=None, known_only=False):
    """Parse queryctx dataset arguments.

    Args:
        args_list: optional list of command-line strings.
        known_only: if True, returns (args, remaining) via parse_known_args.

    Returns:
        argparse.Namespace (or (args, remaining) if known_only).
    """
    # Use a copy to avoid mutating the shared parser
    import copy
    p = copy.copy(_parser)
    if known_only:
        return p.parse_known_args(args_list)
    return p.parse_args(args_list)


def object_args(args_list=None):
    """Get args as a plain Namespace object. Alias for queryctx_args()."""
    import copy
    return copy.copy(_parser).parse_args(args_list)
