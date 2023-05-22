export CUDA_VISIBLE_DEVICES=0

# 事前にdata_2021, data_2022に過去データをunzipし保存する
# 過去データの特徴量作成
python utils/make_sound_dataset.py --sound_dir ../data_2021/train_short_audio --save_dir ../dataset_2021 --meta_path ../data_2021/train_metadata.csv --duration 10
python utils/make_sound_dataset.py --sound_dir ../data_2022/train_audio --save_dir ../dataset_2022 --meta_path ../data_2022/train_metadata.csv --duration 10
python utils/make_sound_dataset.py --sound_dir ../data_2020/train_audio --save_dir ../dataset_2020 --meta_path ../data_2020/train_extended.csv --duration 10
# 鳥なしデータ生成
python utils/make_sound_dataset_soundscape.py
# 2023データの分割
python utils/get_val_files.py
# 2020, 21,22データの分割
python utils/get_val_files_pretrain.py --csv_dir ../csv_2020_2021_2022
# 2021, 22事前学習の設定作成
python utils/make_config.py --batch_size 32 --lr_max 1e-3 --lr_min 1e-5  --max_epoch 50 50 --model bird_sed_eca bird_sed_eca --do_mixup True --training_year 2020_2021_2022 2020_2021_2022  --aug_ver 4 4 --duration 10 10 --loss_fn bce_clip_max_v2_nocall_100 bce_clip_max_v2_nocall_100 --n_split 2 2 #bce_clip_max_v2_nocall_100 bce_clip_max_v2_nocall_100 

pretrain_files="../result/*bird_sed_eca-max_epoch=50-training_year=2020_2021_2022-loss_fn=bce_clip_max_v2_nocall_100-aug_ver=4-duration=10-n_split=2*"
for pretrain_filepath in $pretrain_files; do
    if [ -d $pretrain_filepath ] ; then
        flag="${pretrain_filepath}/fold0_final_model.prm"
        if [ -e $flag ] ; then
            continue
        fi
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
        # 事前学習結果を使用して2023データで再度学習
        python utils/make_config.py --batch_size 32 --lr_max 1e-3 --lr_min 1e-5  --max_epoch 40 --model bird_sed_eca bird_sed_eca --do_mixup True --aug_ver 4 4 --model_path $pretrain_filepath $pretrain_filepath --loss_fn bce_clip_max_v2_nocall_100 bce_clip_max_v2_nocall_100  --duration 10 10
        files="../result/*model_path=$pretrain_filepath*"
        for filepath in $files; do
            if [ -d $filepath ] ; then
                flag="${filepath}/fold4_final_model.prm"
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
