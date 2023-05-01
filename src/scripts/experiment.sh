export CUDA_VISIBLE_DEVICES=1

python utils/get_val_files.py
python utils/make_config.py --batch_size 32 32 --lr_max 1e-4 --lr_min 1e-5  --max_epoch 15 --model bird_base bird_base 

files="../result/*"
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