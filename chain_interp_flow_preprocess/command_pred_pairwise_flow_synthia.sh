



gpu=0
batch_size=1
model=model_concat_upsa
data=../semantic_seg_synthia/processed_pc
model_path=log_train/model.ckpt
num_point=16384
output_dir=../semantic_seg_synthia/init_flow

python pred_pairwise_flow.py \
    --gpu $gpu \
    --model $model \
    --data $data \
    --model_path $model_path \
    --num_point $num_point \
    --batch_size $batch_size \
    --output_dir $output_dir \
    > log_pred_pairwise_flow.txt 2>&1 &

