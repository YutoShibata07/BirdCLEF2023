export CUDA_VISIBLE_DEVICES=1
python utils/get_val_files.py
python utils/get_val_files_pretrain.py

python utils/make_config.py --batch_size 256 --lr_max 1e-3 --lr_min 1e-5  --max_epoch 10 15 --model bird_base --do_mixup True --training_year 2021_2022 2021_2022 


pretrain_files="../result/*training_year=2021_2022*"
for pretrain_filepath in $pretrain_files; do
    if [ -d $pretrain_filepath ] ; then
        echo $pretrain_filepath
        flag="${pretrain_filepath}/final_model.prm"
        if [ -e $flag ] ; then
            continue
        fi
        flag2="${pretrain_filepath}/config.yaml"
        if [ ! -e $flag2 ] ; then
            continue
        fi
        echo $pretrain_filepath
        python train.py "${pretrain_filepath}/config.yaml" --use_wandb
        pretrain_filepath=${pretrain_filepath##*/}
        python utils/make_config.py --batch_size 256 --lr_max 1e-4 1e-3 --lr_min 1e-5  --max_epoch 25 30 --model bird_base --do_mixup True --model_path $pretrain_filepath $pretrain_filepath
        files="../result/*model_path=$pretrain_filepath*"
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
    fi
done
