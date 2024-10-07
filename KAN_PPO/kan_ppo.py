import os
import glob
import time
from datetime import datetime
from .buffer import RolloutBuffer
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import numpy as np
from .kan import KANLayer

class KanActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(KanActorCritic, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(self.device)

        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                            KANLayer(state_dim, 16, 32),
                            KANLayer(32, 16, action_dim)
                        )
        else:
            self.actor = nn.Sequential(
                            KANLayer(state_dim, 16, 32),
                            KANLayer(32, 16, action_dim),
                            nn.Softmax(dim=-1)
                        )
            
        self.critic = nn.Sequential(
                        KANLayer(state_dim, 16, 32),
                        KANLayer(32, 16, 8),
                        KANLayer(8, 4, 1)
                    )