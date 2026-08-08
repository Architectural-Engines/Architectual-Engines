import torch

class AnnealConfig:
    def __init__(self , time_feat_dim = 11 , hidden_dim = 3,  embed_dim = 3 , state_dim = 3, branch_embed_dim = 12, in_dim = 3 ):

        self.embed_dim = embed_dim

        self.in_dim = in_dim

        self.branch_embed_dim = branch_embed_dim

        self.time_feat_dim = time_feat_dim

        self.hidden_dim = hidden_dim

        self.state_dim = state_dim