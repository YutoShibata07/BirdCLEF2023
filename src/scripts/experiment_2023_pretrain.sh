export CUDA_VISIBLE_DEVICES=2
python utils/make_sound_dataset.py --duration 5
python utils/get_val_files.py
python utils/make_config.py --batch_size 16 16 --lr_max 1e-3 --lr_min 1e-5  --max_epoch 40 40 --n_split 5 5 --model bird_maxpool bird_maxpool --do_mixup True  --aug_ver 4 --loss_fn clip_bce clip_bce --duration 5 5 #--model_path exp090_sed_2020_2021_2022 exp090_sed_2020_2021_2022 
# 鳥なしデータ生成
python utils/make_sound_dataset_soundscape.py
files="../result/*maxpool*"
for filepath in $files; do
    if [ -d $filepath ] ; then
        flag="${filepath}/final_model.prm"
        if [ -e $flag ] ; then
            continue
        fi
        flag="${filepath}/fold4_final_model.prm"
        if [ -e $flag ] ; then
            continue
        fi
        flag2="${filepath}/config.yaml"
        if [ ! -e $flag2 ] ; then
            continue
        fi
        # 事前学習はexperiment_w_pretrain.shで実行する
        if [[ $filepath == *"training_year"* ]]; then
            continue
        fi
        echo $filepath
        python train.py "${filepath}/config.yaml" --use_wandb
    fi
done