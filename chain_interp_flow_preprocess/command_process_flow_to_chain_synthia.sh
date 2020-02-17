



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

