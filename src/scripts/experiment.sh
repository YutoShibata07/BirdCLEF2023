export CUDA_VISIBLE_DEVICES=0

python utils/get_val_files.py
python utils/make_config.py --batch_size 16 --lr_max 1e-3 --lr_min 1e-5  --max_epoch 30 --model bird_sed --do_mixup True True

files="../result/*do_mixup*"
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