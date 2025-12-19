import numpy as np
import torch 
from CAESAR.compressor import CAESAR

from dataset import ScientificDataset
from torch.utils.data import DataLoader

data_arg = {
    "data_path": "../data/Turb_Rot_testset.npz",
    # Load dimension 0 at index 8 (e.g. a variable or channel)
    "variable_idx": [0],
    # Load dimension 1 in the range [0:1] (e.g. a spatial or slice range)
    "section_range": [0, 1],
    # Load dimension 2 in the range [0:48] (e.g. frame/timestep range)
    "frame_range": [0, 16],
    # Number of frames per sample: 8 for CAESAR-V, 16 for CAESAR-D
    "n_frame": 8,
}

compressor = CAESAR(model_path="../data/caesar_v.pt", use_diffusion = False, device = "cpu", gae_device = "cpu")

dataset_org = ScientificDataset(data_arg)
shape = dataset_org.data_input.shape
print('dataset shape: ', shape)

dataloader =  DataLoader(dataset_org, batch_size=64, shuffle=False)

# Compression with relative error bound guarantees
compressed_data, compressed_size = compressor.compress(dataloader, eb=0.01)
all_q_latents = [batch["q_latent"] for batch in compressed_data["latent"]]
print('latents batch0 shape, type, max, min: ', all_q_latents[0].shape, all_q_latents[0].dtype, torch.max(all_q_latents[0]), torch.min(all_q_latents[0]))

# Decompression
recons_data = compressor.decompress(compressed_data)

original_data = dataset_org.input_data()
recons_data = dataset_org.recons_data(recons_data)
nrmse = torch.sqrt(torch.mean((original_data - recons_data) ** 2)) / (torch.max(original_data) - torch.min(original_data))
cr = np.prod(original_data.shape)*8/compressed_size
print("NRMSE: ", nrmse.item(), "CR: ", cr.item())

