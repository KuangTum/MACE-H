import torch
import os
from e3nn import o3
from .macehmodules import MACE, NodeDegreeExpansion, EdgeUpdate
from typing import Any, Callable, Dict, List, Optional, Type, Tuple
from torch_geometric.data import Batch
from .e3modules import SeparateWeightTensorProduct, e3LayerNorm
from .statistics import shift_scale_out


# TODO: rmax 需要根据数据集求得?
# deleted keyword: 
class Net(torch.nn.Module):
    def __init__(self, num_species, irreps_edge_init, irreps_sh,
            irreps_post_node, irreps_mid_edge, irreps_post_edge, irreps_out_edge, 
            num_blocks, r_max, use_sc=True, edge_upd=True, num_basis=128, 
            act={1: torch.nn.functional.silu, -1: torch.tanh},
            act_gates={1: torch.sigmoid, -1: torch.tanh},
            num_bessel=8, num_polynomial_cutoff=5, max_ell=3, hidden_irreps='128x0e+128x1o',
            avg_num_neighbors=None, atomic_numbers=None, correlation=3, radial_MLP: Optional[List[int]] = None, mace_norm: str = 'e3LayerNorm',
            expand_nonlin: bool = True, expand_norm: str = 'e3LayerNorm', basis_func: str = 'Bessel', num_gaussian: int = 128,
            shift_scale: bool = True):
        
        super(Net, self).__init__()

        irreps_edge_init = o3.Irreps(irreps_edge_init)
        assert irreps_edge_init == o3.Irreps(f'{irreps_edge_init.dim}x0e')
        irreps_sh =o3.Irreps(irreps_sh)
        irreps_post_node=o3.Irreps(irreps_post_node)
        irreps_mid_edge=o3.Irreps(irreps_mid_edge)
        irreps_post_edge=o3.Irreps(irreps_post_edge)
        irreps_out_edge=o3.Irreps(irreps_out_edge)
        hidden_irreps=o3.Irreps(hidden_irreps)

        self.register_buffer("atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64))
        self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.float64))

        self.num_species = num_species

        if max_ell is None:
            max_ell = irreps_sh.lmax

        self.mace = MACE(r_max=r_max, num_bessel=num_bessel, num_polynomial_cutoff=num_polynomial_cutoff, max_ell=max_ell, num_interactions=num_blocks,
                         num_species=num_species, hidden_irreps=hidden_irreps, avg_num_neighbors=avg_num_neighbors, correlation=correlation,
                         radial_MLP=radial_MLP, mace_norm=mace_norm, basis_func=basis_func, num_gaussian=num_gaussian)

        self.expansion = NodeDegreeExpansion(hidden_irreps=hidden_irreps, max_ell=max_ell, irreps_post_node=irreps_post_node,
                                            num_interactions=num_blocks, num_species=num_species, use_sc=use_sc, expand_nonlin=expand_nonlin,
                                            expand_norm=expand_norm)

        node_irreps_out_list = [block.irreps_out for block in self.expansion.node_order_expansion_blocks]

        self.edge_update = EdgeUpdate(num_species=num_species, irreps_edge_init=irreps_edge_init, irreps_sh=irreps_sh,
                                      node_irreps_out_list=node_irreps_out_list, irreps_mid_edge=irreps_mid_edge,
                                      irreps_post_edge=irreps_post_edge, irreps_out_edge=irreps_out_edge, r_max=r_max,
                                      use_sc=use_sc, num_basis=num_basis, edge_upd=edge_upd,
                                      act=act, act_gates=act_gates)
        
        self.shift_scale = shift_scale
        if shift_scale:
            self.register_buffer('mean_tensor', torch.zeros(num_species, num_species, irreps_out_edge.dim))
            self.register_buffer('std_tensor', torch.ones(num_species, num_species, irreps_out_edge.dim))
        
    def forward(self, data: Batch):

        edge_attr = data["edge_attr"][:, [0, 2, 3, 1]] # (y, z, x) order

        node_one_hot = torch.nn.functional.one_hot(data.x, num_classes=self.num_species).type(torch.get_default_dtype())
        
        node_feats_hidden_list, node_feats_up = self.mace(edge_attr, data.edge_index, node_one_hot, data.batch)

        post_node_feats_list = self.expansion(node_feats_hidden_list, node_feats_up, node_one_hot, data.batch)

        node_fea, edge_fea = self.edge_update(post_node_feats_list, edge_attr, data.edge_index, data.x, data.batch)

        if self.shift_scale:
            edge_fea = shift_scale_out(edge_fea, self.mean_tensor, self.std_tensor, data.x, data.edge_index) #, self.edge_update.irreps_out_edge)

        return node_fea, edge_fea
    
    def __repr__(self):
        info = '===== MACEH model structure: ====='
        info += f'\nusing spherical harmonics in MACE: {self.mace.spherical_harmonics.irreps_out}'
        info += f'\nusing spherical harmonics in EdgeUpdate: {self.edge_update.irreps_sh}'
        for index, (nupd, eupd) in enumerate(zip(self.expansion.node_order_expansion_blocks, self.edge_update.edge_update_blocks)):
            info += f'\n=== layer {index} ==='
            info += f'\nnode update: ({self.mace.interactions[0].hidden_irreps} -> {nupd.irreps_out})'
            if eupd is not None:
                info += f'\nedge update: ({eupd.irreps_in_edge} -> {eupd.irreps_out})'
        info += '\n=== output ==='
        info += f'\noutput edge: ({self.edge_update.irreps_out_edge})'
        
        return info
    
    def analyze_tp(self, path):
        os.makedirs(path, exist_ok=True)
        for index, (ninter, nexpand, eupd) in enumerate(zip(self.mace.interactions, self.expansion.node_order_expansion_blocks,
                                                    self.edge_update.edge_update_blocks)):
            fig, ax = ninter.conv_tp.visualize()
            fig.savefig(os.path.join(path, f'node_update_{index}.png'))
            fig.clf()
            fig, ax = nexpand.tp.tp.visualize()
            fig.savefig(os.path.join(path, f'node_expand_{index}.png'))
            fig.clf()
            fig, ax = eupd.conv.tp.tp.visualize()
            fig.savefig(os.path.join(path, f'edge_update_{index}.png'))
            fig.clf()