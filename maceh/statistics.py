import torch 
from e3nn import o3
from .data import AijData
from .e3modules import e3TensorDecomp

# def get_mean_std_tensor(dataset: AijData, net_out_irreps: o3.Irreps) -> tuple:
#     # config.net_out_irreps a.k.a irreps_out_edge

#     assert 'label' in dataset._data.keys(), "label is not found in the dataset"

#     atomic_number_edge_i = dataset._data.x[dataset._data.edge_index[0]]
#     atomic_number_edge_j = dataset._data.x[dataset._data.edge_index[1]]
#     # we have to leave the 'source to target' order in pyg alone, since deeph use this order

#     num_species = dataset.info['index_to_Z'].numel()
#     mean_tensor = torch.zeros(num_species, num_species, net_out_irreps.dim).to(dtype=dataset._data.label.dtype, device=dataset._data.label.device)
#     std_tensor = torch.ones(num_species, num_species, net_out_irreps.dim).to(dtype=dataset._data.label.dtype, device=dataset._data.label.device)
#     for x_i in range(num_species):
#          for x_j in range(num_species):
#               label_ij = dataset._data.label[torch.logical_and(atomic_number_edge_i==x_i, atomic_number_edge_j==x_j)]
#               for l, sli in zip(net_out_irreps.ls, net_out_irreps.slices()):
#                    if l == 0:
#                        mean_tensor[x_i, x_j, sli] = label_ij[:, sli].mean()
#                    std_tensor[x_i, x_j, sli] = label_ij[:, sli].std()
   
#     return mean_tensor, std_tensor



# def get_mean_std_tensor(dataset: AijData, net_out_irreps: o3.Irreps) -> tuple:
#     # config.net_out_irreps a.k.a irreps_out_edge

#     assert 'label' in dataset._data.keys(), "label is not found in the dataset"

#     atomic_number_edge_i = dataset._data.x[dataset._data.edge_index[0]]
#     atomic_number_edge_j = dataset._data.x[dataset._data.edge_index[1]]
#     # we have to leave the 'source to target' order in pyg alone, since deeph use this order

#     num_species = dataset.info['index_to_Z'].numel()
#     mean_tensor = torch.zeros(num_species, num_species, net_out_irreps.dim).to(dtype=dataset._data.label.dtype, device=dataset._data.label.device)
#     std_tensor = torch.ones(num_species, num_species, net_out_irreps.dim).to(dtype=dataset._data.label.dtype, device=dataset._data.label.device)
#     for x_i in range(num_species):
#          for x_j in range(num_species):
#               label_ij = dataset._data.label[torch.logical_and(atomic_number_edge_i==x_i, atomic_number_edge_j==x_j)]
#               for l, sli in zip(net_out_irreps.ls, net_out_irreps.slices()):
#                    if l == 0:
#                        mean_tensor[x_i, x_j, sli] = label_ij[:, sli].mean()
#                    std_tensor[x_i, x_j, sli] = label_ij[:, sli].std()
   
#     return mean_tensor, std_tensor



def get_mean_std_tensor(dataset: AijData, net_out_irreps: o3.Irreps, kernel: e3TensorDecomp) -> tuple:
    # config.net_out_irreps a.k.a irreps_out_edge

    assert 'label' in dataset._data.keys(), "label is not found in the dataset"

    net_out = kernel.get_net_out(dataset._data.label.to(kernel.device)).cpu()

    atomic_number_edge_i = dataset._data.x[dataset._data.edge_index[0]]
    atomic_number_edge_j = dataset._data.x[dataset._data.edge_index[1]]
    # we have to leave the 'source to target' order in pyg alone, since deeph use this order

    num_species = dataset.info['index_to_Z'].numel()
    mean_tensor = torch.zeros(num_species, num_species, net_out_irreps.dim).to(dtype=net_out.dtype, device=net_out.device)
    std_tensor = torch.ones(num_species, num_species, net_out_irreps.dim).to(dtype=net_out.dtype, device=net_out.device)
    for x_i in range(num_species):
         for x_j in range(num_species):
              net_out_ij = net_out[torch.logical_and(atomic_number_edge_i==x_i, atomic_number_edge_j==x_j)]
              for l, sli in zip(net_out_irreps.ls, net_out_irreps.slices()):
                   if l == 0:
                       mean_tensor[x_i, x_j, sli] = net_out_ij[:, sli].mean()
                   std_tensor[x_i, x_j, sli] = net_out_ij[:, sli].std()

    # std_tensor = torch.where(std_tensor<1., std_tensor, torch.ones_like(std_tensor))

    # # std_tensor = torch.where(std_tensor < std_tensor.quantile(0.5, dim=-1, keepdim=True), std_tensor, torch.tile(std_tensor.quantile(0.5, dim=-1, keepdim=True), dims=(1,1,std_tensor.shape[2])))
 
    return mean_tensor, std_tensor



# def shift_scale_out(edge_feature: torch.Tensor, mean_tensor: torch.Tensor, std_tensor: torch.Tensor, x: torch.Tensor, edge_index: torch.LongTensor,
#                     net_out_irreps: o3.Irreps = None) -> torch.Tensor:
     
#     atomic_number_edge_i = x[edge_index[0]]
#     atomic_number_edge_j = x[edge_index[1]]
#     # we have to leave the 'source to target' order in pyg alone, since deeph use this order

#     num_species = mean_tensor.shape[0]

#     for x_i in range(num_species):
#          for x_j in range(num_species):
#             mask = torch.logical_and(atomic_number_edge_i==x_i, atomic_number_edge_j==x_j)
#             #   for l, sli in zip(net_out_irreps.ls, net_out_irreps.slices()):
#             #     if l == 0:
#             #         edge_feature[mask][:, sli] = edge_feature[mask][:, sli] * std_tensor[x_i, x_j, sli] + mean_tensor[x_i, x_j, sli]
#             #     else:
#             #         edge_feature[mask][:, sli] = edge_feature[mask][:, sli] * std_tensor[x_i, x_j, sli]
#             edge_feature[mask] = edge_feature[mask] * std_tensor[x_i, x_j] + mean_tensor[x_i, x_j]
         
#     return edge_feature



def shift_scale_out(edge_feature: torch.Tensor, mean_tensor: torch.Tensor, std_tensor: torch.Tensor, x: torch.Tensor, edge_index: torch.LongTensor,
                    ) -> torch.Tensor:
     
    atomic_number_edge_i = x[edge_index[0]]
    atomic_number_edge_j = x[edge_index[1]]
    # we have to leave the 'source to target' order in pyg alone, since deeph use this order

    num_species = mean_tensor.shape[0]

    for x_i in range(num_species):
         for x_j in range(num_species):
            mask = torch.logical_and(atomic_number_edge_i==x_i, atomic_number_edge_j==x_j)
            #   for l, sli in zip(net_out_irreps.ls, net_out_irreps.slices()):
            #     if l == 0:
            #         edge_feature[mask][:, sli] = edge_feature[mask][:, sli] * std_tensor[x_i, x_j, sli] + mean_tensor[x_i, x_j, sli]
            #     else:
            #         edge_feature[mask][:, sli] = edge_feature[mask][:, sli] * std_tensor[x_i, x_j, sli]

            # edge_feature[mask] = edge_feature[mask] * std_tensor[x_i, x_j] + mean_tensor[x_i, x_j]

            edge_feature[mask] = edge_feature[mask] + (edge_feature[mask] * (std_tensor[x_i, x_j] - 1.)).detach() + mean_tensor[x_i, x_j] 
         
    return edge_feature



# def inverse_shift_scale_out(edge_feature: torch.Tensor, mean_tensor: torch.Tensor, std_tensor: torch.Tensor, x: torch.Tensor, edge_index: torch.LongTensor,
#                             net_out_irreps: o3.Irreps = None) -> torch.Tensor:
     
#     atomic_number_edge_i = x[edge_index[0]]
#     atomic_number_edge_j = x[edge_index[1]]
#     # we have to leave the 'source to target' order in pyg alone, since deeph use this order

#     num_species = mean_tensor.shape[0]

#     for x_i in range(num_species):
#          for x_j in range(num_species):
#             mask = torch.logical_and(atomic_number_edge_i==x_i, atomic_number_edge_j==x_j)
#             #   for l, sli in zip(net_out_irreps.ls, net_out_irreps.slices()):
#             #     if l == 0:
#             #         edge_feature[mask][:, sli] = (edge_feature[mask][:, sli] - mean_tensor[x_i, x_j, sli]) / std_tensor[x_i, x_j, sli]
#             #     else:
#             #         edge_feature[mask][:, sli] = edge_feature[mask][:, sli] / std_tensor[x_i, x_j, sli]
#             edge_feature[mask] = (edge_feature[mask] - mean_tensor[x_i, x_j]) / std_tensor[x_i, x_j]
         
#     return edge_feature



def inverse_shift_scale_out(edge_feature: torch.Tensor, mean_tensor: torch.Tensor, std_tensor: torch.Tensor, x: torch.Tensor, edge_index: torch.LongTensor,
                            ) -> torch.Tensor:
     
    atomic_number_edge_i = x[edge_index[0]]
    atomic_number_edge_j = x[edge_index[1]]
    # we have to leave the 'source to target' order in pyg alone, since deeph use this order

    num_species = mean_tensor.shape[0]

    for x_i in range(num_species):
         for x_j in range(num_species):
            mask = torch.logical_and(atomic_number_edge_i==x_i, atomic_number_edge_j==x_j)
            #   for l, sli in zip(net_out_irreps.ls, net_out_irreps.slices()):
            #     if l == 0:
            #         edge_feature[mask][:, sli] = (edge_feature[mask][:, sli] - mean_tensor[x_i, x_j, sli]) / std_tensor[x_i, x_j, sli]
            #     else:
            #         edge_feature[mask][:, sli] = edge_feature[mask][:, sli] / std_tensor[x_i, x_j, sli]
            edge_feature[mask] = (edge_feature[mask] - mean_tensor[x_i, x_j]) / std_tensor[x_i, x_j]
         
    return edge_feature
