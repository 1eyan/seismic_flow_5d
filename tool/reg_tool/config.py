import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpuIndex', type=int, default=6)
parser.add_argument('--model_type', type=str, default='gated')
parser.add_argument(
    '--infer_mode',
    type=str,
    default='fpm',
    choices=['fpm', 'e2e'],
    help='推理范式：fpm=FlowMatching采样，e2e=端到端前向',
)
parser.add_argument('--use_missing_focus_adapter', type=bool, default=False)
parser.add_argument('--geom_mode', type=str, default='source')
parser.add_argument('--checkpoint_path', type=str, default='/home/chengzhitong/5d_regular/seis_flow_data12V2/resultsFPM/gated_seisdit_datatype_dongfang_gated_seisdit_Linear_velocity_missing_ratio0.4/checkpoints/model-50.pth')
# 规则化重建流程：数据路径与输出
parser.add_argument('--irregular_h5_file', type=str, default='/NAS/czt/mount/seis_flow_data12V2/h5_/C3NA_01_irregular.h5', help='非规则观测 H5 文件')
parser.add_argument('--regular_h5_file', type=str, default='/NAS/czt/mount/seis_flow_data12V2/h5_/C3NA_01.h5', help='规则网格 H5 文件（含 shot_line）')
parser.add_argument('--output_dir', type=str, default='/home/chengzhitong/5d_regular/seis_flow_data12V2/reg_results', help='重建结果与可视化输出目录')
parser.add_argument('--window_size', type=int, default=10, help='滑动窗口道数')
parser.add_argument('--target_size', type=int, default=128, help='每窗目标道数（邻居+规则道）')
parser.add_argument('--max_shots', type=int, default=None, help='最多处理炮数，None 表示全部')
parser.add_argument('--max_lines_per_shot', type=int, default=None, help='每炮最多测线数，None 表示全部')
parser.add_argument('--test_training', action='store_true', help='从训练集加载样本做预测并可视化，不跑规则化重建')
parser.add_argument('--test_training_num', type=int, default=3, help='测试时使用的训练样本数量')
parser.add_argument('--test_training_indices', type=str, default=None, help='逗号分隔的样本索引，如 0,5,10；默认随机取')
# Dataset 推理模式（与 DatasetH5_all train=False 对齐）：patch 级推理与可视化
parser.add_argument('--infer_max_patches', type=int, default=None, help='最多推理 patch 数，None 表示全部')
parser.add_argument('--infer_indices', type=str, default=None, help='逗号分隔的 patch 索引；不指定则按 infer_max_patches 或全量')
parser.add_argument('--legacy_shot_line', action='store_true', help='使用旧版 KDTree+shot/line 规则化重建（需 irregular/regular H5）')
args, _ = parser.parse_known_args()