# 협업학습기술(ETRI) 관련연구 

## Summary
In the distributed setting, we study weight update algorithms(like DSGD, ADMM, and Collaborative Learning). This code leads a brief overview of ways in which we can solve this problem. 

## Install
```python
pip install -r requirement.txt
```

## Quickstart
```python
python main.py ./data ./output/ring1 --epochs 200 --batch_size 100 --l2_lambda 0.01 --model_name resnet50 --dataset_name cifar10 --group_channels 32 --drop_rate 0.1 --last_drop_rate 0.5 --loglevel INFO --train_data_length 10000 --cuda_device_no 1 --skip_plots --scheduler StepLR --step_size 410 --gamma 0.9 --sleep_factor 0 DSGD node0 ./conf/node_list.json ./conf/hosts.json --lr 0.002 --swap_timeout 10 --async_step > ./output/ring1/node0.txt
python main.py ./data ./output/ring2 --epochs 200 --batch_size 100 --l2_lambda 0.01 --model_name resnet50 --dataset_name cifar10 --group_channels 32 --drop_rate 0.1 --last_drop_rate 0.5 --loglevel INFO --train_data_length 10000 --cuda_device_no 2 --skip_plots --scheduler StepLR --step_size 410 --gamma 0.9 --sleep_factor 0 DSGD node1 ./conf/node_list.json ./conf/hosts.json --lr 0.002 --swap_timeout 10 --async_step > ./output/ring2/node1.txt 
python main.py ./data ./output/ring3 --epochs 200 --batch_size 100 --l2_lambda 0.01 --model_name resnet50 --dataset_name cifar10 --group_channels 32 --drop_rate 0.1 --last_drop_rate 0.5 --loglevel INFO --train_data_length 10000 --cuda_device_no 3 --skip_plots --scheduler StepLR --step_size 410 --gamma 0.9 --sleep_factor 0 DSGD node2 ./conf/node_list.json ./conf/hosts.json --lr 0.002 --swap_timeout 10 --async_step > ./output/ring3/node2.txt 
```
```python
 parser = ArgumentParser()
 parser.add_argument("datadir", type=str, help="dataset path")
 parser.add_argument("outdir", type=str, help="output path")

 parser.add_argument("--epochs", type=int, default=1000, help="epochs")
 parser.add_argument("--batch_size", type=int, default=64, help="batch size")
 parser.add_argument("--seed", type=int, default=569, help="seed")
 parser.add_argument("--data_init_seed", type=int,
                        default=11, help="data init seed")
 parser.add_argument("--model_init_seed", type=int,
                        default=13, help="model init seed")
 parser.add_argument("--l2_lambda", type=float, default=0.001,
                        help="L2 lambda")
 parser.add_argument("--model_name", type=str, default="resnet50",
                        choices=["logistics", "resnet50"])
 parser.add_argument("--dataset_name", type=str,
                        default="cifar10")
 parser.add_argument("--train_data_length", type=int, default=12800,
                        help="train_data_length / batch_size = 1epoch = n round")
 parser.add_argument("--group_channels", type=int, default=32)
 parser.add_argument("--drop_rate", type=float, default=0.1,
                        help="dropout層のdropout rate")
 parser.add_argument("--last_drop_rate", type=float, default=0.5,
                        help="dropout rate")
 parser.add_argument("--cpu", dest="cuda", default=True, action="store_false",
                        help="CPU")
 parser.add_argument("--cuda_device_no", type=int, default=0)
 parser.add_argument("--skip_plots", dest="plot", default=True, action="store_false",
                        help="skip plots")

 parser.add_argument("--scheduler", type=str,
                        default="none", choices=["none", "StepLR"])
 parser.add_argument("--step_size", type=int, default=5)
 parser.add_argument("--gamma", type=float, default=0.90)

 parser.add_argument("--sleep_factor", type=float, default=0.0)

 parser.add_argument("--loglevel",
                        type=lambda x: LogLevel.name_of(x),
                        default=LogLevel.DEBUG,
                        help="Log level")
 parser.add_argument("--logfile", type=str, default=None,
                        help="Log file path")

 optim_parsers = parser.add_subparsers(title="optimizer", dest="optimizer")
 optim = optim_parsers.add_parser("sgd")
 optim.add_argument("--lr", type=float, default=0.002)
 optim.add_argument("--momentum", type=float, default=0.0)
 optim.add_argument("--dampening", type=float, default=0.0)
 optim.add_argument("--weight_decay", type=float, default=0.0)
 optim.add_argument("--nesterov", default=False, action="store_true")

 optim = optim_parsers.add_parser("adam")
 optim.add_argument("--lr", type=float, default=0.002)
 optim.add_argument("--betas", type=float, default=(0.9, 0.999), nargs=2)
 optim.add_argument("--eps", type=float, default=1e-8)
 optim.add_argument("--weight_decay", type=float, default=0.0)
 optim.add_argument("--amsgrad", default=False, action="store_true")
    
 optim = optim_parsers.add_parser("PdmmISVR")
 optim.add_argument("nodename", type=str)
 optim.add_argument("conf", type=str)
 optim.add_argument("host", type=str)
 optim.add_argument("--lr", type=float, default=0.002)
 optim.add_argument("--use_gcoef", default=False, action="store_true")
 optim.add_argument("--piw", type=float, default=1.0,
                       choices=[0.5, 1.0, 2.0])
 optim.add_argument("--async_step", dest="round_step",
                       default=True, action="store_false")
 optim.add_argument("--swap_timeout", type=int, default=100)
    
 optim = optim_parsers.add_parser("AdmmISVR")
 optim.add_argument("nodename", type=str)
 optim.add_argument("conf", type=str)
 optim.add_argument("host", type=str)
 optim.add_argument("--lr", type=float, default=0.002)
 optim.add_argument("--use_gcoef", default=False, action="store_true")
 optim.add_argument("--piw", type=float, default=1.0,
                     choices=[0.5, 1.0, 2.0])
 optim.add_argument("--async_step", dest="round_step",
                       default=True, action="store_false")
 optim.add_argument("--swap_timeout", type=int, default=1000)
    
 optim = optim_parsers.add_parser("DSGD")
 optim.add_argument("nodename", type=str)
 optim.add_argument("conf", type=str)
 optim.add_argument("host", type=str)
 optim.add_argument("--lr", type=float, default=0.002)
 optim.add_argument("--momentum", type=float, default=0.0)
 optim.add_argument("--dampening", type=float, default=0.0)
 optim.add_argument("--weight_decay", type=float, default=0.0)
 optim.add_argument("--nesterov", default=False, action="store_true")
 optim.add_argument("--weight", type=float, default=1.0)
 optim.add_argument("--async_step", dest="round_step",
                       default=True, action="store_false")
 optim.add_argument("--swap_timeout", type=int, default=1)
```

## Distributed Deep Learing: From Single-Node to P2P(Peer-to-Peer)

### Single-Node Training (with SGD)
```python
epoch,   train_loss,    val_loss,    test_loss,    train_acc,     val_acc,     test_acc
1,            2.616,       2.192,        2.191,        0.173,       0.178,        0.175
...
200,          0.311,       0.187,        0.998,        0.942,       0.939,        0.721
```

### P2P(Peer-to-Peer) Training (with DSGD)
```python
(Node-1)
epoch,   train_loss,    val_loss,    test_loss,    train_acc,     val_acc,     test_acc
1,            2.560,       2.417,        2.415,        0.142,       0.111,        0.118
...
200,           1.761,      1.617,        1.630,        0.465,       0.424,        0.416  
(Node-2)
epoch,   train_loss,    val_loss,    test_loss,    train_acc,     val_acc,     test_acc
1,            2.580,       2.349,        2.349,        0.186,       0.148,        0.145
...
200,          1.654,       1.624,        1.632,        0.462,       0.426,        0.421
(Node-3)
epoch,   train_loss,    val_loss,    test_loss,    train_acc,     val_acc,     test_acc
1,            2.555,       2.389,        2.390,        0.186,       0.121,        0.119
...
100,          1.632,       1.687,        1.701,        0.490,       0.406,        0.405  
...
200,
```

### P2P(Peer-to-Peer) Training (with ADMM)
