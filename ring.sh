python main.py ./data ./output --epochs 100 --batch_size 100 --l2_lambda 0.01 --model_name logistics --dataset_name cifar10 --group_channels 32 --drop_rate 0.1 --last_drop_rate 0.5 --loglevel INFO --train_data_length 10000 --cpu --skip_plots --scheduler StepLR --step_size 410 --gamma 0.9 --sleep_factor 0 DSGD node0 ./conf/node_list.json ./conf/hosts.json --lr 0.002 --swap_timeout 10 --async_step > ./output/node0.txt 2>&1 &
python main.py ./data ./output --epochs 100 --batch_size 100 --l2_lambda 0.01 --model_name logistics --dataset_name cifar10 --group_channels 32 --drop_rate 0.1 --last_drop_rate 0.5 --loglevel INFO --train_data_length 10000 --cpu --skip_plots --scheduler StepLR --step_size 410 --gamma 0.9 --sleep_factor 0 DSGD node1 ./conf/node_list.json ./conf/hosts.json --lr 0.002 --swap_timeout 10 --async_step > ./output/node1.txt 2>&1 &
python main.py ./data ./output --epochs 100 --batch_size 100 --l2_lambda 0.01 --model_name logistics --dataset_name cifar10  --group_channels 32 --drop_rate 0.1 --last_drop_rate 0.5 --loglevel INFO --train_data_length 10000 --cpu  --skip_plots --scheduler StepLR --step_size 410 --gamma 0.9 --sleep_factor 0 DSGD node2 ./conf/node_list.json ./conf/hosts.json --lr 0.002 --swap_timeout 10 --async_step > ./output/node2.txt 2>&1 &

