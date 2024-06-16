# Last version
# This code shows the creative method that merges the ViT-base and the ViT-Huge, so-called level-wise ViT



import torch
import torch.nn as nn
from IPython import embed
import math
from utils import utils_transform


nn.Module.dump_patches = True




class EgoExo4D(nn.Module):
    def __init__(self, input_dim, output_dim, num_layer, embed_dim, nhead, device,opt):
        super(EgoExo4D, self).__init__()

        self.linear_embedding = nn.Linear(input_dim,embed_dim)
# The low_level encoder uses the ViT-base
        low_level_encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead = 8)
# The high_level encoder uses the ViT-Huge
        high_level_encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead = 16)
        self.low_level_transformer_encoder = nn.TransformerEncoder(low_level_encoder_layer, num_layers = 3)

        self.high_level_transformer_encoder = nn.TransformerEncoder(high_level_encoder_layer, num_layers = 32)      

        out_dim =51

        self.low_level_stabilizer = nn.Sequential(
                        nn.Linear(embed_dim, 256),
                        nn.ReLU(),
                        nn.Linear(256, out_dim)
        )
        

        self.high_level_stabilizer = nn.Sequential(
                        nn.Linear(embed_dim, 256),
                        nn.ReLU(),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Linear(128, out_dim)
        )

        


    def forward(self, input_tensor,image=None, do_fk = True):
        input_tensor = input_tensor.reshape(input_tensor.shape[0],input_tensor.shape[1],-1)
        x = self.linear_embedding(input_tensor)
        x = x.permute(1,0,2)
        x1 = self.low_level_transformer_encoder(x)
        x2 = self.high_level_transformer_encoder(x)
        
        x1 = x1.permute(1,0,2)[:, -1]
        x2 = x2.permute(1,0,2)[:, -1]
        low_level_global_orientation = self.low_level_stabilizer(x1)
        high_level_global_orientation = self.high_level_stabilizer(x2)
        
        w1 = 0.1
        w2 = 0.9

        global_orientation = w1 * low_level_global_orientation + w2 * high_level_global_orientation
        return global_orientation
