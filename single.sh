python main.py ./data ./output/single --epochs 200 --batch_size 100 --l2_lambda 0.01 --model_name resnet50 --dataset_name cifar10 --group_channels 32 --drop_rate 0.1 --last_drop_rate 0.5 --loglevel INFO --train_data_length 10000 --cuda_device_no 0  --skip_plots --scheduler StepLR --step_size 410 --gamma 0.9 --sleep_factor 0 sgd --lr 0.002