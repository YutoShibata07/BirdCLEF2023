export CUDA_VISIBLE_DEVICES=1

# python utils/create_roi_image.py --input_dataset train_images_1024
python utils/make_config.py --batch_size 3560 3560 --lr_max 1e-3 --lr_min 1e-4  --max_epoch 15 --model lstm_ver3 lstm_ver3 --file_limit 100 --long_sample aux_and_max aux_and_max --pulse_limit 96 --num_layers 4 4 --embed_dim 196 196

files="../result/*and*"
for filepath in $files; do
    if [ -d $filepath ] ; then
        flag="${filepath}/final_model.prm"
        if [ -e $flag ] ; then
            continue
        fi
        flag2="${filepath}/config.yaml"
        if [ ! -e $flag2 ] ; then
            continue
        fi
        echo $filepath
        python train.py "${filepath}/config.yaml" --use_wandb
    fi
done