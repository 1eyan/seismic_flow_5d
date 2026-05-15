# Dataset configuration for SEG-Y -> H5 conversion.
#
# ---------------------------------------------------------------------------
# Single-file mode (used by batch_segy2h5.py):
#   segyPairs maps group_name -> [input_segy, label_segy, task, method, extra]
#   info_h5: output H5 path
# ---------------------------------------------------------------------------
# info_h5 = '/path/to/output.h5'
# segyPairs = {
#     'group1': ['/data/raw.sgy', '/data/label.sgy', 'interp', '5d_kdtree'],
# }

# ---------------------------------------------------------------------------
# Triple-file mode (used by Segy2H5.py --irr/--mask/--label):
#   Run from command line:
#     python Segy2H5.py --irr /data/irregular.sgy --mask /data/mask.sgy \
#         --label /data/label.sgy --dataset-name field1031 --mode self_computed
#
#   Output structure:
#     <common_root>/h5/<dataset_name>_irregular.h5
#     <common_root>/h5/<dataset_name>_mask.h5
#     <common_root>/h5/<dataset_name>_label.h5
#
#   Or call convert_segy_triple() from Python:
#     from Segy2H5 import convert_segy_triple
#     convert_segy_triple('/data/irr.sgy', '/data/mask.sgy', '/data/label.sgy',
#                         dataset_name='field1031', mode='self_computed')
# ---------------------------------------------------------------------------

# Example: single-file config (current active)
# info_h5 = '../h5/sw06_mask_70.h5'
# segyPairs = {
#     '1551': [
#         '/NAS/czt/mount/chengzhitong/data/dongfang_syn_reg/mask_miss70pct_004-sw06-Sj5-label.sgy',
#         '/NAS/czt/mount/chengzhitong/data/dongfang_syn_reg/mask_miss70pct_004-sw06-Sj5-label.sgy',
#         'interp',
#         '5d_line_by_order',
#         'none'
#     ]
# }
