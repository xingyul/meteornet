


command_file=`basename "$0"`
gpu=0
model=model_part_seg_meteor_direct
data=processed_pc
num_point=8192
num_frame=3
max_epoch=150
batch_size=12
learning_rate=0.0016
# model_path=log_${model}_labelweights_1.2_new_radius_step_1/model-17.ckpt
model_path=None
log_dir=log_${model}


python train_direct.py \
    --gpu $gpu \
    --data $data \
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
