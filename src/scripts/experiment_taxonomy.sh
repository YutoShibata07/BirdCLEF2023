export CUDA_VISIBLE_DEVICES=1

python utils/get_taxonomy_dict.py 
python utils/get_val_files.py
python utils/make_config.py --batch_size 8 --lr_max 1e-3 --lr_min 1e-5  --max_epoch 30 --model bird_taxonomy bird_taxonomy --do_mixup True  --aug_ver 4 4 --loss_fn focal_clip_max_taxonomy focal_clip_max_taxonomy --duration 10 10 --model_path exp102_sed_taxonomy_2023 exp102_sed_taxonomy_2023
# 鳥なしデータ生成
python utils/make_sound_dataset_soundscape.py
files="../result/*exp102_sed_taxonomy_2023*"
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