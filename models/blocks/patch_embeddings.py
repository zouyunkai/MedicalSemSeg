from functools import reduce
from operator import mul

import functorch
import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import to_3tuple

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
    def __init__(self, patch_size, out_dim, vector_dim, intensity_transform=None, patch_position_embeddings=True, final_layer=True):

        super().__init__()

        self.patch_size = to_3tuple(patch_size)
        self.final_layer = final_layer
        self.patch_position_embeddings = patch_position_embeddings
        self.vector_dim = vector_dim
        self.out_dim = out_dim
        self.voxels_per_patch = reduce(mul, self.patch_size)


        if self.final_layer:
            self.fc = nn.Linear(self.voxels_per_patch*self.vector_dim, self.out_dim)
        else:
            assert self.voxels_per_patch*self.vector_dim == self.out_dim

        if self.patch_position_embeddings:
            tot_patch_size = self.voxels_per_patch
        else:
            tot_patch_size = 1

        if not intensity_transform is None:
            self.intensity_intervals = torch.from_numpy(intensity_transform(HU_INTENSITY_INTERVALS)).cuda()
        else:
            self.intensity_intervals = torch.from_numpy(HU_INTENSITY_INTERVALS).cuda()
        self.n_interval_points = len(self.intensity_intervals)

        self.vectors = []
        for i in range(len(self.intensity_intervals)):
            patch_vectors = []
            for j in range(tot_patch_size):
                patch_vectors.append(nn.Parameter(torch.randn(vector_dim)))
            self.vectors.append(patch_vectors)

    def forward(self, x):
        B, C, D, H, W = x.size()
        Pd, Ph, Pw = self.patch_size

        patches = x.unfold(2, Pd, Pd).unfold(3, Ph, Ph).unfold(4, Pw, Pw)
        patches = patches.contiguous().view(-1, Pd, Ph, Pw)

        patch_fun = functorch.vmap(self.create_patch_vectors, randomness='same')
        voxel_vectors = patch_fun(patches) # n_patches, (Pd * Ph * Pw), vector_dim

        patch_vectors = voxel_vectors.flatten(1)

        if self.final_layer:
            patch_vectors = self.fc(patch_vectors)

        assert patch_vectors.shape[1] == self.out_dim

        patch_vectors = patch_vectors.view(B, D // Pd, H // Ph, W // Pw, self.out_dim)

        patch_vectors = patch_vectors.contiguous().permute(0, 4, 1, 2, 3)

        return patch_vectors

    def create_patch_vectors(self, patch):
        Pd, Ph, Pw = patch.size()
        assert self.patch_size[0] == Pd and self.patch_size[1] == Ph and self.patch_size[2] == Pw

        patch = patch.view(-1)

        vecs = []
        if self.patch_position_embeddings:
            for i, p in enumerate(patch):
                v = self.get_voxel_vector(p, i)
                vecs.append(v)
        else:
            for p in patch:
                v = self.get_voxel_vector(p)
                vecs.append(v)

        patch_vectors = torch.stack(v, dim=0)

        return patch_vectors

    def get_voxel_vector(self, voxel, voxel_index=0):

        if voxel >= self.intensity_intervals[-1]:
            indx = self.n_interval_points - 2
            weight = 1.0
        elif voxel <= self.intensity_intervals[0]:
            indx = 0
            weight = 0.0
        else:
            tmp = torch.cat([self.intensity_intervals, voxel.unsqueeze()])
            _, indices = tmp.sort()
            indx = (indices == self.n_interval_points).nonzero(as_tuple=True)[0] - 1
            ints_range = self.intensity_intervals[indx + 1] - self.intensity_intervals[indx]
            weight = (voxel - self.intensity_intervals[indx]) / ints_range

        vector_a = self.vectors[indx][voxel_index]
        vector_b = self.vectors[indx + 1][voxel_index]
        vector_out = weight*vector_b + (1-weight)*vector_a

        return vector_out

    def hu_intensity_to_index_and_weight(self, intensity):
        indx = np.searchsorted(self.intensity_intervals, intensity.item())
        ints_range = self.intensity_intervals[indx] - self.intensity_intervals[indx-1]
        weight = (intensity - self.intensity_intervals[indx-1]) / ints_range
        return indx, weight


