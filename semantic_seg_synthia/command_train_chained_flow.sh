


command_file=`basename "$0"`
gpu=0
model=model_part_seg_meteor_chained_flow
data=processed_pc
data_chained_flow=chained_flow
num_point=8192
num_frame=2
max_epoch=150
batch_size=8
learning_rate=0.001
model_path=None
log_dir=log_${model}_frame_${num_frame}


python train_chained_flow.py \
    --gpu $gpu \
    --data $data \
    --data_chained_flow $data_chained_flow \
    --model $model \
    --learning_rate $learning_rate \
    --model_path $model_path \
    --log_dir $log_dir \
    --num_point $num_point \
    --num_frame $num_frame \
    --max_epoch $max_epoch \
    --batch_size $batch_size \
    --command_file $command_file \
    > $log_dir.txt 2>&1 &
