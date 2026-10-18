"""RADAR decoder boundary for E1 AGDA action-key/bias residuals.

The native SVD/Sinkhorn encoder and native glimpse are retained. Host feature
packing and the native training loop are intentionally outside this boundary.
The adapter must be installed BEFORE optimizer construction. No long-run
launcher is enabled until environment/feature/replay acceptance is complete.
"""
import torch
from torch import nn
from caliroute.plugins.agda import AdaptiveGraphAttention


class RadarAGDADecoder(nn.Module):
    def __init__(self, native_decoder, *, enabled=False, candidate_feature_dim=30,
                 system_feature_dim=15):
        super().__init__()
        self.native = native_decoder
        self.enabled = bool(enabled)
        width = native_decoder.model_params['embedding_dim']
        self.agda = AdaptiveGraphAttention(width, enabled=True,
            enable_delta_k=False, enable_delta_v=False,
            enable_delta_action_key=True, enable_action_bias=True,
            optimize_dynamic_projections=True, use_agda_v2=False,
            candidate_feature_dim=candidate_feature_dim,
            system_feature_dim=system_feature_dim)
        self.node_embeddings = None
        self.context = None
        self.activation_checkpointing = False
        self.trajectory_chunk_size = None

    def set_kv(self, encoded_jobs):
        # Rebuilt on every native pre_forward, including after an optimizer step.
        self.node_embeddings = encoded_jobs
        self.context = None
        return self.native.set_kv(encoded_jobs)

    def set_q1(self, encoded_q1):
        return self.native.set_q1(encoded_q1)

    def forward(self, encoded_q0, load, ninf_mask, *, node_embeddings=None,
                decision_tokens=None, candidate_features=None, system_features=None):
        if not self.enabled:
            return self.native(encoded_q0, load, ninf_mask)
        if self.context is not None and candidate_features is None:
            node_embeddings = self.node_embeddings
            decision_tokens, candidate_features, system_features = self.context
        if any(x is None for x in (node_embeddings, decision_tokens, candidate_features, system_features)):
            raise ValueError('Enabled AGDA requires explicit host features and tokens')
        batch, trajectories, nodes = ninf_mask.shape
        if candidate_features.shape[:3] != (batch, trajectories, nodes):
            raise ValueError('Candidate features must match native batch/trajectory/node order')
        if node_embeddings.shape[:2] != (batch, nodes):
            raise ValueError('Node embedding order/shape mismatch')
        chunk = self.trajectory_chunk_size or trajectories
        if chunk < 1:
            raise ValueError('trajectory_chunk_size must be positive')
        outputs=[]
        # Native environment mutates load and mask during env.step. Checkpoint
        # backward must replay the decision-time state, not the final state.
        if self.activation_checkpointing and torch.is_grad_enabled():
            load = load.clone()
            ninf_mask = ninf_mask.clone()
        for start in range(0,trajectories,chunk):
            stop=min(start+chunk,trajectories)
            args=(encoded_q0[:,start:stop],load[:,start:stop],ninf_mask[:,start:stop],
                  node_embeddings,decision_tokens[:,start:stop],candidate_features[:,start:stop],system_features[:,start:stop])
            if self.activation_checkpointing and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                outputs.append(checkpoint(self._enabled_forward,*args,use_reentrant=False,preserve_rng_state=True))
            else:
                outputs.append(self._enabled_forward(*args))
        return outputs[0] if len(outputs)==1 else torch.cat(outputs,dim=1)

    def _enabled_forward(self, encoded_q0, load, ninf_mask, node_embeddings,
                         decision_tokens, candidate_features, system_features):
        batch, trajectories, nodes = ninf_mask.shape
        _, _, action_key, bias = self.agda(node_embeddings, decision_tokens,
            candidate_features, system_features=system_features)
        native = self.native
        heads = native.model_params['head_num']
        query = native.Wq_0(torch.cat((encoded_q0, load[..., None]), -1))
        query = query.reshape(batch, trajectories, heads, -1).transpose(1, 2)
        glimpse = native.multi_head_combine(native._multi_head_attention(
            query, native.k, native.v, rank3_ninf_mask=ninf_mask))
        # Retain the native action key and add only a candidate-conditioned delta.
        scores = torch.matmul(glimpse, native.single_head_key)
        scores = scores + torch.einsum('btd,btnd->btn', glimpse, action_key)
        scaled = scores / native.model_params['sqrt_embedding_dim']
        # Match E1's calc_logits: the pointer clips its scaled dot-product,
        # then adds AGDA bias. The preparation prototype put bias inside tanh.
        clipped = native.model_params['logit_clipping'] * torch.tanh(scaled) + bias
        return torch.softmax(clipped + ninf_mask, dim=-1)


def install_radar_agda(model, *, enabled=True):
    """CVRP-only integration; install BEFORE native trainer creates optimizer.

    Nine candidate features: forward/reverse road row, candidate->depot,
    depot->candidate (all divided by instance max road km), demand/capacity,
    remaining load, visited-customer, feasible, depot indicator. Four system
    features: remaining load, unserved demand/capacity, unserved fraction,
    current->depot/max km. No invented time or battery resource inputs.
    Tokens: current node, depot, graph mean, mean unvisited customer embedding.
    This is a host-specific feature schema, not the E1 30/15 schema.
    """
    if isinstance(model.decoder, RadarAGDADecoder):
        raise ValueError('AGDA already installed')
    native = model.decoder
    device = next(model.parameters()).device
    # The plugin must not shift host initialization or random SVD/action draws.
    devices = [device.index or 0] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        model.decoder = RadarAGDADecoder(native, enabled=enabled,
            candidate_feature_dim=9, system_feature_dim=4).to(device)
    original_pre = model.pre_forward
    def pre(reset):
        model._e2_road = reset.problems
        model._e2_demand = torch.cat((reset.node_demand.new_zeros(reset.node_demand.size(0), 1), reset.node_demand), -1)
        return original_pre(reset)
    model.pre_forward = pre
    def context_hook(module, args):
        state = args[0]
        if state.selected_count < 2 or not module.decoder.enabled:
            return
        h = module.encoded_node
        road = module._e2_road
        b, s = state.current_node.shape
        n = road.size(-1)
        bi = torch.arange(b, device=h.device)[:, None]
        current = state.current_node
        scale = road.amax((-1, -2)).clamp_min(1e-6)[:, None, None]
        forward = road[bi, current] / scale
        reverse = road.transpose(-1, -2)[bi, current] / scale
        depot_to = (road[:, 0, :][:, None] / scale).expand(b, s, n)
        to_depot = (road[:, :, 0][:, None] / scale).expand(b, s, n)
        demand = module._e2_demand[:, None].expand(b, s, n)
        # Native ninf mask mixes capacity and visitation; reconstruct visitation
        # from the action history in a hook below, never infer it from that mask.
        visited = module._e2_visited
        feasible = torch.isfinite(state.ninf_mask)
        depot = torch.zeros_like(demand); depot[..., 0] = 1
        remaining = (~visited).to(h.dtype); remaining[..., 0] = 0
        candidate = torch.stack((forward, reverse, to_depot, depot_to, demand,
            state.load[..., None].expand_as(demand), visited.to(h.dtype),
            feasible.to(h.dtype), depot), -1)
        system = torch.stack((state.load, (demand * remaining).sum(-1),
            remaining.sum(-1) / (n-1), forward[..., 0]), -1)
        mean_unvisited = torch.einsum('bsn,bnd->bsd', remaining, h) / remaining.sum(-1, keepdim=True).clamp_min(1)
        tokens = torch.stack((h[bi, current], h[:, 0, None].expand(b, s, -1),
            h.mean(1)[:, None].expand(b, s, -1), mean_unvisited), -2)
        module.decoder.context = (tokens, candidate, system)
    def record_hook(module, args, output):
        state = args[0]
        action = output[0]
        if state.selected_count == 0:
            module._e2_visited = torch.zeros((*action.shape, module._e2_road.size(-1)), dtype=torch.bool, device=action.device)
        # Replace, don't mutate a tensor saved in the previous step's autograd.
        module._e2_visited = module._e2_visited.scatter(-1, action[..., None], True)
    model.register_forward_pre_hook(context_hook)
    model.register_forward_hook(record_hook)
    return model
