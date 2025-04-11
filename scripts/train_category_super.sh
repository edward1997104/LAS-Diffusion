# please modify the sdf_folder to your own path

# singe category
python train_super_resolution.py --data_class Thingi10K --name shape_super --batch_size 4 --new True --continue_training False --training_epoch 500  --split_dataset True --sdf_folder /data/octfusion_data/Thingi10K/sdf