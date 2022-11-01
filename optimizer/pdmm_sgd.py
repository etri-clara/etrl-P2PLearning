import logging

import torch

from connection.contract import Contract

__all__ = ["PdmmSGD"]


class PdmmSGD(Contract):
    def __init__(self, name, round_cnt, edges, hosts, model, device="cpu",
                 lr=0.002, momentum=0, dampening=0, weight_decay=0, nesterov=False, round_step=False, weight=1.0,
                 swap_timeout=10):
        mu = 200
        eta = 1.0
        eta_rate = eta / mu
        defaults = dict(lr=lr, eta=eta, rho=rho, initial_lr=lr, eta_rate=eta_rate)
        super(PdmmSGD, self).__init__(name, round_cnt, edges, hosts, model, defaults, device, round_step, weight,
                                   is_dual=False, swap_timeout=swap_timeout)

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
        edges = self.edges()
        edge_num = len(edges) + 1

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
                    consensus += vs_metric_eta / edge_num * edge.prm_a * edge.prm_dual["rcv"][i] 
                    
                    if self._is_state:
                        proximity += vs_metric_rho / edge_num * edge.prm_state["rcv"][i]
                        coefficient += vs_metric_eta / edge_num + vs_metric_rho / edge_num
                    else:
                        coefficient += vs_metric_eta / edge_num

                p.data = (v_grad * p_data - m_grad + consensus + proximity) / coefficient

                for edge in edges:
                    edge.prm_state["snd"][i] = p.data
                

        self.swap_params("state")
        self.round_update()
    
    @torch.no_grad()
    def diff(self):
        
        for edge in self.edges():
            diff_buf = edge.diff_buff()
            if diff_buf is not None:
                buf_name_list = list(diff_buf)
                for group in self.param_groups:
                    for i, p in enumerate(group['params']):
                        self._diff += self._criterion(p.data, diff_buf[buf_name_list[i]])

        return self._diff