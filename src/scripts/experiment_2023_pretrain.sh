export CUDA_VISIBLE_DEVICES=1

python utils/get_val_files.py
python utils/make_config.py --batch_size 32 --lr_max 1e-3 --lr_min 1e-5  --max_epoch 30 --model bird_sed_b1 bird_sed_b1 --do_mixup True  --aug_ver 4 4 --loss_fn focal_clip_max_v2 --duration 10 10 --n_split 5 5 #--model_path exp090_sed_2020_2021_2022 exp090_sed_2020_2021_2022
# 鳥なしデータ生成
python utils/make_sound_dataset_soundscape.py
files="../result/*n_split*"
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
        # 事前学習はexperiment_w_pretrain.shで実行する
        if [[ $filepath == *"training_year"* ]]; then
            continue
        fi
        echo $filepath
        python train.py "${filepath}/config.yaml" --use_wandb
    fi
done