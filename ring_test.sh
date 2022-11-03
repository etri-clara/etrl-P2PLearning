python main.py ./data ./output/ring1 --epochs 200 --batch_size 100 --l2_lambda 0.01 --model_name resnet50 --dataset_name cifar10 --group_channels 32 --drop_rate 0.1 --last_drop_rate 0.5 --loglevel INFO --train_data_length 10000 --cuda_device_no 1 --skip_plots --scheduler StepLR --step_size 410 --gamma 0.9 --sleep_factor 0 AdmmSGD node0 ./conf/node_list.json ./conf/hosts.json --lr 0.002 --swap_timeout 10 --async_step > ./output/ring1/node1.txt 2>&1 &


python main.py ./data ./output/ring2 --epochs 200 --batch_size 100 --l2_lambda 0.01 --model_name resnet50 --dataset_name cifar10 --group_channels 32 --drop_rate 0.1 --last_drop_rate 0.5 --loglevel INFO --train_data_length 10000 --cuda_device_no 2 --skip_plots --scheduler StepLR --step_size 410 --gamma 0.9 --sleep_factor 0 AdmmSGD node1 ./conf/node_list.json ./conf/hosts.json --lr 0.002 --swap_timeout 10 --async_step > ./output/ring2/node2.txt 2>&1 &



python main.py ./data ./output/ring3 --epochs 200 --batch_size 100 --l2_lambda 0.01 --model_name resnet50 --dataset_name cifar10 --group_channels 32 --drop_rate 0.1 --last_drop_rate 0.5 --loglevel INFO --train_data_length 10000 --cuda_device_no 3 --skip_plots --scheduler StepLR --step_size 410 --gamma 0.9 --sleep_factor 0 AdmmSGD node2 ./conf/node_list.json ./conf/hosts.json --lr 0.002 --swap_timeout 10 --async_step > ./output/ring3/node3.txt 2>&1 &
