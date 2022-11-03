import logging

import torch

from connection.contract import Contract

__all__ = ["GossipSGD"]


class GossipSGD(Contract):
    def __init__(self, name, round_cnt, edges, hosts, model, device="cpu",
                 lr=0.002, momentum=0, dampening=0, weight_decay=0, nesterov=False, round_step=False, weight=1.0,
                 swap_timeout=10):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay,
                        nesterov=nesterov, initial_lr=lr)
        super(GossipSGD, self).__init__(name, round_cnt, edges, hosts, model, defaults, device, round_step, weight,
                                   is_dual=False, swap_timeout=swap_timeout)
        logging.info(f"Optimizer {type(self)} params: {defaults}")

    def __setstate__(self, state):
        super(GossipSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        edges = self.edges()
        edge_nums = len(edges)

        for edge in edges:
            for group in self.param_groups:
                for i, p in enumerate(group['params']):
                    d_p = p.data
                    p.data = torch.div((d_p + edge.prm_state["rcv"][i]), 2)
                    
                    edge.prm_state["snd"][i] = p.data
                    

        self.swap_params("state")
        self.round_update()