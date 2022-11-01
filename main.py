import os
import logging
import json
import gc
import numpy as np
from argparse import ArgumentParser
from util.log import LogLevel, config_logger


os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)

if __name__ == "__main__":
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
                        help="最後のdropout層のdropout rate")
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

    optim = optim_parsers.add_parser("AdmmSGD")
    optim.add_argument("nodename", type=str)
    optim.add_argument("conf", type=str)
    optim.add_argument("host", type=str)
    optim.add_argument("--lr", type=float, default=0.002)
    optim.add_argument("--mu", type=int, default=200)
    optim.add_argument("--eta", type=float, default=1.0)
    optim.add_argument("--eta_rate", type=float, default=1.0)
    optim.add_argument("--async_step", dest="round_step",
                       default=True, action="store_false")
    optim.add_argument("--swap_timeout", type=int, default=1)


    args = parser.parse_args()

    config_logger(loglevel=args.loglevel, logfile=args.logfile)
    logging.info(args)


    optim_args = {}
    if args.optimizer == "sgd":
        optim_args["lr"] = args.lr
        optim_args["momentum"] = args.momentum
        optim_args["weight_decay"] = args.weight_decay
        optim_args["dampening"] = args.dampening
        optim_args["nesterov"] = args.nesterov
    elif args.optimizer == "adam":
        optim_args["lr"] = args.lr
        optim_args["betas"] = args.betas
        optim_args["eps"] = args.eps
        optim_args["weight_decay"] = args.weight_decay
        optim_args["amsgrad"] = args.amsgrad
    elif args.optimizer == "PdmmISVR":
        optim_args["lr"] = args.lr
        optim_args["round_step"] = args.round_step
        optim_args["use_gcoef"] = args.use_gcoef
        optim_args["drs"] = False
        optim_args["piw"] = args.piw
        optim_args["swap_timeout"] = args.swap_timeout
    elif args.optimizer == "AdmmISVR":
        optim_args["lr"] = args.lr
        optim_args["round_step"] = args.round_step
        optim_args["use_gcoef"] = args.use_gcoef
        optim_args["drs"] = True
        optim_args["piw"] = args.piw
        optim_args["swap_timeout"] = args.swap_timeout
    elif args.optimizer == "DSGD":
        optim_args["lr"] = args.lr
        optim_args["momentum"] = args.momentum
        optim_args["weight_decay"] = args.weight_decay
        optim_args["dampening"] = args.dampening
        optim_args["nesterov"] = args.nesterov
        optim_args["weight"] = args.weight
        optim_args["round_step"] = args.round_step
        optim_args["swap_timeout"] = args.swap_timeout
    elif args.optimizer == "AdmmSGD":
        optim_args["lr"] = args.lr

    scheduler_args = []
    scheduler_kwargs = {}
    if args.scheduler == "none":
        pass
    elif args.scheduler == "StepLR":
        scheduler_args.append(args.step_size)
        scheduler_kwargs["gamma"] = args.gamma

    if args.optimizer in ["sgd", "adam"]:
        from util.trainer import Trainer
        trainer = Trainer(args.outdir, args.seed,
                          datadir=args.datadir,
                          dataset_name=args.dataset_name,
                          model_name=args.model_name,
                          group_channels=args.group_channels,
                          drop_rate=args.drop_rate, last_drop_rate=args.last_drop_rate,
                          data_init_seed=args.data_init_seed, model_init_seed=args.model_init_seed,
                          cuda=args.cuda, cuda_device_no=args.cuda_device_no)

    elif args.optimizer in ["PdmmISVR", "AdmmISVR", "DSGD", "AdmmSGD"]:
        from util.dist_trainer import DistTrainer as Trainer

        with open(args.conf) as f:
            conf = json.load(f)
        logging.info(conf)
        nodes = conf["nodes"][args.nodename]
        edges = nodes["edges"]
        optim_args["round"] = nodes["round"]
        with open(args.host) as f:
            hosts = json.load(f)["hosts"]

        nodeidx = sorted([n["name"] for n in hosts]).index(args.nodename)
        np.random.seed(args.seed)
        seed = np.random.randint(0, 25485227, len(hosts))[nodeidx]

        trainer = Trainer(args.outdir,
                          args.nodename, edges, hosts,
                          seed,
                          datadir=args.datadir,
                          dataset_name=args.dataset_name,
                          model_name=args.model_name,
                          group_channels=args.group_channels,
                          drop_rate=args.drop_rate, last_drop_rate=args.last_drop_rate,
                          data_init_seed=args.data_init_seed, model_init_seed=args.model_init_seed,
                          train_data_length=args.train_data_length,
                          cuda=args.cuda,
                          cuda_device_no=args.cuda_device_no)
        trainer.sleep_factor = args.sleep_factor
    else:
        raise ValueError("Unknown optimizer: %s" % (args.optimizer))

    trainer.train(args.epochs, args.batch_size,
                  optimizer=args.optimizer, optim_args=optim_args,
                  scheduler_name=args.scheduler, scheduler_args=scheduler_args, scheduler_kwargs=scheduler_kwargs,
                  l2_lambda=args.l2_lambda)
    trainer.evaluate(args.batch_size)
    trainer.dispose()
    trainer.save_state_dict()
    if args.plot:
        trainer.plot_figs()
    del trainer
    gc.collect()
    logging.info("GC: check garbage %s" % (str(gc.garbage)))
    gc.collect()


# Usage
# python main.py ./data ./output sgd
