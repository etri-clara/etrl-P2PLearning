import logging
import math
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from connection.contract import Contract

__all__ = ["AdmmSGD"]


class AdmmSGD(Contract):
    def __init__(self, name, round_cnt, edges, hosts, model, device="cpu",
                 lr=0.002, mu=200, eta=1.0, rho=0.1, round_step=False, weight=1.0,
                 swap_timeout=10):
        mu = 200
        eta = 1.0
        eta_rate = eta / mu
        rho = 0.1

        self._is_state = True
        if rho == 0:
            self._is_state = False
        
        defaults = dict(lr=lr, mu=mu, eta=eta, rho=rho, initial_lr=lr, eta_rate=eta_rate)
        super(AdmmSGD, self).__init__(name, round_cnt, edges, hosts, model, defaults, device, round_step, weight,
                                  is_avg = True, swap_timeout=swap_timeout)


        m_state = model.state_dict()
        dim_num_ary = []
        prev_dim_num = 1
        for name in m_state:
            dim_num = 1
            if name.endswith(".weight"):
                if m_state[name].ndim > 1:
                    for i, dim in enumerate(m_state[name].shape):
                        # except out_dim
                        if i > 0:
                            dim_num *= dim
                else:
                    # for GrpNorm
                    dim_num *= m_state[name].shape[0]
                prev_dim_num = dim_num
            else:
                # bias
                dim_num = prev_dim_num
            dim_num_ary.append(dim_num)

        for group in self.param_groups:
            group["dim_num"] = dim_num_ary

        logging.info(f"Optimizer {type(self)} params: {defaults}")

    def __setstate__(self, state):
        super(PdmmSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        edges = self.edges()
        edge_num = len(edges)

        for group in self.param_groups:
            mu = 1 / group["lr"]
            group["eta"] = mu * group["eta_rate"]

            for i, p in enumerate(group['params']):

                vs_metric = math.sqrt(group["dim_num"][i])
                consensus = torch.zeros_like(p)
                proximity = torch.zeros_like(p)

                p_data = p.data
                m_grad = p.grad.data
                v_grad = torch.zeros_like(p)
                vs_metric_eta = vs_metric * group["eta"]
                vs_metric_rho = vs_metric * group["rho"]
                torch.nn.init.constant_(v_grad, mu)
                coefficient = v_grad.clone()

                for edge in edges:
                    # admm
                    # consensus += vs_metric_eta / edge_num * edge.prm_a() * edge.dual_avg(i)
                    # -> consensus += vs_metric_eta / edge_num * edge.prm_a * edge.dual_avg[i]
                    # pdmm
                    # consensus += vs_metric_eta / edge_num * edge.prm_a() * edge.rcv_dual()[i]
                    # -> consensus += vs_metric_eta / edge_num * edge.prm_a * edge.prm_dual["rcv"][i] 
                    consensus += vs_metric_eta / edge_num * edge.prm_a * edge.dual_avg[i]
                    
                    if self._is_state:
                        proximity += vs_metric_rho / edge_num * edge.prm_state["rcv"][i]
                        coefficient += vs_metric_eta / edge_num + vs_metric_rho / edge_num
                    else:
                        coefficient += vs_metric_eta / edge_num

                p.data = (v_grad * p_data - m_grad + consensus + proximity) / coefficient

                for edge in edges:
                    edge.prm_state["snd"][i] = p.data
                    edge.prm_dual["snd"][i] = edge.prm_dual["rcv"][i] - \
                        2 * edge.prm_a * p.data
                

        self.swap_params("dual")
        self.round_update()

        if closure is not None:
            loss = closure()
        return loss