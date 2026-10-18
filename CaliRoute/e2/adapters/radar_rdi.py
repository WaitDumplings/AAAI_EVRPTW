"""Extra directed-road attention bias, retaining RADAR SVD and Sinkhorn."""
import torch
from torch import nn
from caliroute.plugins.rdi import RoadDistanceInjection


class CVRPRoadInjection(RoadDistanceInjection):
    def features(self, distance, *args, **kwargs):
        features = super().features(distance)
        # CVRP has no time or battery resources. Keep the portable core's
        # layout, but explicitly disable all six non-distance input channels.
        mask = features.new_tensor([1,1,1,0,0,0,0,0,1,0])
        return features * mask


class RadarRDIAttention(nn.Module):
    def __init__(self, native, *, enabled=True):
        super().__init__()
        self.native = native
        self.enabled = enabled
        self.road_bias = None

    def forward(self, q, k, v, cost_mat):
        if not self.enabled:
            return self.native(q, k, v, cost_mat)
        if self.road_bias is None:
            raise RuntimeError('RDI context requires model.pre_forward first')
        # Same native mixed score operations and same ten Sinkhorn iterations.
        # The sole insertion is the zero-initialized bias before Sinkhorn.
        from ACVRPModel_LIB import sinkhorn_normalization
        native = self.native
        b, heads, rows, _ = q.shape
        cols = k.size(2)
        dot = torch.matmul(q, k.transpose(2,3)) / native.model_params['sqrt_qkv_dim']
        road = cost_mat[:,None].expand(b,heads,rows,cols)
        features = torch.stack((dot, road, road.transpose(2,3)), dim=4).transpose(1,2)
        hidden = torch.relu(torch.matmul(features,native.mix1_weight)+native.mix1_bias[None,None,:,None,:])
        scores = (torch.matmul(hidden,native.mix2_weight)+native.mix2_bias[None,None,:,None,:]).transpose(1,2).squeeze(4)
        weights = sinkhorn_normalization(scores + self.road_bias)
        return torch.matmul(weights,v).transpose(1,2).reshape(b,rows,heads*native.model_params['qkv_dim'])


def install_radar_rdi(model, *, enabled=True):
    if hasattr(model,'rdi'):
        raise ValueError('RDI already installed')
    device=next(model.parameters()).device
    with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
        model.rdi=CVRPRoadInjection(model.model_params['head_num']).to(device)
    for layer in model.encoder.layers:
        layer.mixed_score_MHA=RadarRDIAttention(layer.mixed_score_MHA,enabled=enabled)
    original=model.pre_forward
    def pre(reset):
        # Rebuild the learned bias every batch/weight version. Native SVD and
        # native distance standardization remain inside original_pre_forward.
        bias=model.rdi(reset.problems) if enabled else None
        for layer in model.encoder.layers:layer.mixed_score_MHA.road_bias=bias
        return original(reset)
    model.pre_forward=pre
    return model
