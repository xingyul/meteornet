


parser.add_argument('--original_pc_dir', default='../semantic_seg_synthia/processed_pc', help='Original point cloud directory [default: ../
semantic_seg_synthia/processed_pc]')
parser.add_argument('--pairwise_flow_dir', default='../semantic_seg_synthia/init_flow', help='Initial scene flow directory [default: ../
semantic_seg_synthia/init_flow]')
parser.add_argument('--chained_flowed_dir', default='../semantic_seg_synthia/chained_flow', help='Chained scene flow directory, output
[default: ../semantic_seg_synthia/chained_flow]')
parser.add_argument('--maximum_frame_diff', type=int, default=3, help='Maximum frame difference [default: 3]')
parser.add_argument('--num_threads', type=int, default=4, help='Number of threads [default: 3]')


original_pc_dir=../semantic_seg_synthia/processed_pc
pairwise_flow_dir=../semantic_seg_synthia/init_flow
chained_flowed_dir=../semantic_seg_synthia/chained_flow
maximum_frame_diff=3
num_threads=4
num_nn=2

python process_flow_to_chain.py \
    --original_pc_dir $original_pc_dir \
    --pairwise_flow_dir $pairwise_flow_dir \
    --chained_flowed_dir $chained_flowed_dir \
    --maximum_frame_diff $maximum_frame_diff \
    --num_threads $num_threads \
    --num_nn $num_nn \
    > log_process_flow_to_chain.txt 2>&1 &

