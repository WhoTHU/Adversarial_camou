import torch
import itertools
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def grid_sample(input, grid, canvas = None):
    output = F.grid_sample(input, grid)
    if canvas is None:
        return output
    else:
        input_mask = Variable(input.data.new(input.size()).fill_(1))
        output_mask = F.grid_sample(input_mask, grid)
        padded_output = output * output_mask + canvas * (1 - output_mask)
        return padded_output

class TPSGridGen(nn.Module):
    def __init__(self, target_shape=None, target_control_points=None, target_coordinate=None):
        super(TPSGridGen, self).__init__()
        self.target_shape = target_shape

        assert target_control_points.ndimension() == 2
        self.ndim = target_control_points.size(1)
        N = target_control_points.size(0)
        self.num_points = N
        target_control_points = target_control_points.float()
        self.register_buffer('target_control_points', target_control_points)

        # create padded kernel matrix
        forward_kernel = torch.zeros(N + 1 + self.ndim, N + 1 + self.ndim)
        target_control_partial_repr = self.compute_partial_repr(target_control_points, target_control_points)
        forward_kernel[:N, :N].copy_(target_control_partial_repr)
        forward_kernel[:N, N].fill_(1)
        forward_kernel[N, :N].fill_(1)
        forward_kernel[:N, N+1:].copy_(target_control_points)
        forward_kernel[N+1:, :N].copy_(target_control_points.transpose(0, 1))
        
        # compute inverse matrix
        inverse_kernel = torch.inverse(forward_kernel)

        # create target cordinate matrix
        if target_coordinate is None:
            assert self.ndim == 2
            HW = target_shape.numel()
            Y, X = torch.meshgrid(*[torch.linspace(-1, 1, s) for s in target_shape])
            target_coordinate = torch.stack([X.flatten(), Y.flatten()], dim=1) # convert from (y, x) to (x, y)
            target_coordinate_partial_repr = self.compute_partial_repr(target_coordinate, target_control_points)
            target_coordinate_repr = torch.cat([
                target_coordinate_partial_repr, torch.ones(HW, 1), target_coordinate
            ], dim=1)
        else:
            target_coordinate_partial_repr = self.compute_partial_repr(target_coordinate, target_control_points)
            target_coordinate_repr = torch.cat([
                target_coordinate_partial_repr, torch.ones(target_coordinate.shape[0], 1), target_coordinate
            ], dim=1)

        # register precomputed matrices
        self.register_buffer('target_coordinate', target_coordinate.clone().detach())
        self.register_buffer('inverse_kernel', inverse_kernel)
        self.register_buffer('padding_matrix', torch.zeros(self.ndim + 1, self.ndim))
        self.register_buffer('target_coordinate_repr', target_coordinate_repr)

    def forward(self, source_control_points):
        assert source_control_points.ndimension() == 3
        assert source_control_points.size(1) == self.num_points
        assert source_control_points.size(2) == self.ndim
        batch_size = source_control_points.size(0)

        Y = torch.cat([source_control_points, Variable(self.padding_matrix.expand(batch_size, -1, -1))], 1)
        mapping_matrix = torch.matmul(Variable(self.inverse_kernel), Y)
        new_coordinate = torch.matmul(Variable(self.target_coordinate_repr), mapping_matrix)
        return new_coordinate

    # phi(x1, x2) = r^2 * log(r), where r = ||x1 - x2||_2
    def compute_partial_repr(self, input_points, control_points):
        N = input_points.size(0)
        M = control_points.size(0)
        pairwise_diff = input_points.view(N, 1, self.ndim) - control_points.view(1, M, self.ndim)
        pairwise_dist = (pairwise_diff * pairwise_diff).sum(-1)
        repr_matrix = 0.5 * pairwise_dist * pairwise_dist.log()
        # fix numerical error for 0 * log(0), substitute all nan with 0
        mask = repr_matrix != repr_matrix
        repr_matrix.masked_fill_(mask, 0)
        return repr_matrix

    def tps_mesh(self, source_control_points=None, max_range=(0.1, ), batch_size=1):

        if source_control_points is None:
            source_control_points = self.target_control_points.expand(batch_size, -1, -1)
            source_control_points = source_control_points + source_control_points.new(source_control_points.shape).uniform_(-1, 1) * source_control_points.new(max_range)
            # source_control_points = source_control_points.to(self.padding_matrix.device)
        source_coordinate = self.forward(source_control_points)
        return source_coordinate


    def tps_trans(self, inputs, source_control_points=None, max_range=0.1, canvas=0.5, target_shape=None):
        if target_shape is not None:
            if target_shape != self.target_shape:
                device = self.padding_matrix.device
                self.__init__(target_shape, self.target_control_points.cpu())
                self.to(device)

        target_height, target_width = self.target_shape

        if source_control_points is None:
            source_control_points = self.target_control_points.unsqueeze(0) + self.target_control_points.new(
                size=(inputs.shape[0], ) + self.target_control_points.shape).uniform_(-1, 1) * max_range
            # source_control_points = source_control_points.to(self.padding_matrix.device)
        if isinstance(canvas, float):
            canvas = torch.FloatTensor(inputs.shape[0], inputs.shape[1], target_height, target_width).fill_(canvas).to(self.padding_matrix.device)
        source_coordinate = self.forward(source_control_points)
        grid = source_coordinate.view(inputs.shape[0], target_height, target_width, 2)
        target_image = grid_sample(inputs, grid, canvas)
        return target_image, source_control_points