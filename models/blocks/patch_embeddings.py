from functools import reduce
from operator import mul

import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import to_3tuple

from utils.misc import get_1d_sincos_embed_from_range

HU_INTENSITY_INTERVALS = np.array([
                            -1000, # Air
                            -75,   # Fat
                            0,     # Water
                            15,    # Cerebrospinal Fluid
                            25,    # Muscle/Kidney
                            40,    # Blood
                            50,    # Liver
                            200,   # Soft Tissue
                            1000   # Bone
                            ])


class LearnedClassVectors(nn.Module):
    def __init__(self, patch_size, out_dim, vector_dim, intensity_transform=None, static_sincos=False, final_layer=False, concat_vector=False):

        super().__init__()

        self.patch_size = to_3tuple(patch_size)
        self.final_layer = final_layer
        self.vector_dim = vector_dim
        self.out_dim = out_dim
        self.voxels_per_patch = reduce(mul, self.patch_size)
        self.static_sincos = static_sincos
        self.concat_vector = concat_vector

        if not intensity_transform is None:
            self.intensity_intervals = intensity_transform(HU_INTENSITY_INTERVALS)
        else:
            self.intensity_intervals = HU_INTENSITY_INTERVALS
        self.n_intervals = len(self.intensity_intervals) + 1

        if self.final_layer and self.concat_vector:
            assert self.vector_dim == self.n_intervals
            self.fc = nn.Linear(self.vector_dim, self.out_dim)
        elif self.final_layer:
            self.fc = nn.Linear(self.voxels_per_patch * self.vector_dim, self.out_dim)
        else:
            assert self.voxels_per_patch*self.vector_dim == self.out_dim



        self.vectors = nn.ParameterList()
        self.vectors_cls = nn.ParameterList()

        if self.static_sincos:
            sincos_emb = get_1d_sincos_embed_from_range(vector_dim, np.arange(self.n_intervals))

        for i in range(self.n_intervals):
            if self.static_sincos:
                interval_param = nn.Parameter(torch.zeros(self.vector_dim), requires_grad=False)
                interval_param.data.copy_(torch.from_numpy(sincos_emb[i]).float())
                self.vectors.append(interval_param)
            elif self.concat_vector:
                interval_param = nn.Parameter(torch.zeros(self.vector_dim), requires_grad=False)
                interval_param[i] = 1.0
                self.vectors.append(interval_param)
            else:
                self.vectors.append(nn.Parameter(torch.randn(self.vector_dim)))
            self.vectors_cls.append(nn.Parameter((torch.ones(1)*(i+1)*-10000).repeat(self.vector_dim), requires_grad=False))


    def forward(self, x):
        B, C, D, H, W = x.size()
        Pd, Ph, Pw = self.patch_size

        voxel_vectors = self.create_voxel_vectors(x) # n_patches * (Pd * Ph * Pw), vector_dim

        voxel_vectors = voxel_vectors.view(B, C, D, H, W, self.vector_dim).squeeze(1) # B, D, H, W, vector_dim assuming C=1
        voxel_vectors = voxel_vectors.permute(0, 4, 1, 2, 3).contiguous() # B, vector_dim, D, H, W
        patches = voxel_vectors.unfold(2, Pd, Pd).unfold(3, Ph, Ph).unfold(4, Pw, Pw) # B, vector_dim, D/Pd, H/Ph, W/Pw, Pd, Ph, Pw
        if self.concat_vector:
            patch_vectors = patches.sum(-1).sum(-1).sum(-1) # B, vector_dim, D/Pd, H/Ph, W/Pw
            patch_vectors = patch_vectors.permute(0, 2, 3, 4, 1).contiguous() # B, D/Pd, H/Ph, W/Pw, vector_dim
        else:
            patches = patches.permute(0, 2, 3, 4, 5, 6, 7, 1).contiguous() # B, D/Pd, H/Ph, W/Pw, Pd, Ph, Pw, vector_dim
            patch_vectors = patches.flatten(4) # B, D/Pd, H/Ph, W/Pw, (Pd * Ph * Pw * vector_dim)


        if self.final_layer:
            patch_vectors = self.fc(patch_vectors)

        patch_vectors = patch_vectors.permute(0, 4, 1, 2, 3).contiguous()

        return patch_vectors


    def create_voxel_vectors(self, x):
        x = x.flatten(0)
        x = x.view(-1,1)

        x = torch.where(x < self.intensity_intervals[0], self.vectors_cls[0], x)
        for i in range(0, (self.n_intervals - 2)):
            x = torch.where((x >= self.intensity_intervals[i]) & (x < self.intensity_intervals[i+1]), self.vectors_cls[i+1], x)
        x = torch.where(x >= self.intensity_intervals[-1], self.vectors_cls[-1], x)

        for i in range(self.n_intervals):
            x = torch.where(x == (i+1)*-10000, self.vectors[i], x)

        return x