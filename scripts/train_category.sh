# please modify the sdf_folder to your own path

# singe category
python train.py --data_class Thingi10K --name Thingi10K --batch_size 16 --new True --continue_training False --image_size 64 --training_epoch 4000 --ema_rate 0.999 --base_channels 32 --save_last False --save_every_epoch 200 --with_attention True --use_text_condition False --use_sketch_condition False --split_dataset True  --lr 1e-4 --optimizier adamw --sdf_folder data/Thingi10K/sdf