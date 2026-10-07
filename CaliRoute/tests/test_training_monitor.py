import json
import torch
from offline2online.training_monitor import begin_monitor_epoch, module_update_snapshot, finish_module_update, plugin_diagnostics, append_monitor_row, average_diagnostics


class MonitoredModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rdi_adapter = torch.nn.Linear(2, 2, bias=False)
        self.critic = torch.nn.Linear(2, 1, bias=False)


def test_monitor_samples_epoch_one_and_interval_and_measures_real_update():
    model = MonitoredModel()
    assert begin_monitor_epoch(model, 1, {'monitor_interval': 20})
    (model.rdi_adapter.weight.sum() + model.critic.weight.sum()).backward()
    snapshot = module_update_snapshot(model)
    assert set(snapshot) == {'rdi', 'critic'}
    assert module_update_snapshot(model) == {}
    with torch.no_grad():
        model.rdi_adapter.weight.add_(.1)
    finish_module_update(model, snapshot)
    values = plugin_diagnostics(model)
    assert abs(values['module_rdi_grad_norm_unclipped'] - 2.) < 1e-6
    assert abs(values['module_rdi_update_norm'] - .2) < 1e-6
    assert values['module_critic_update_norm'] == 0.
    assert not begin_monitor_epoch(model, 2, {'monitor_interval': 20})
    assert plugin_diagnostics(model) == {}
    assert begin_monitor_epoch(model, 20, {'monitor_interval': 20})


def test_monitor_writes_strict_json_and_preserves_dynamic_keys(tmp_path):
    path = tmp_path / 'monitor_rank_0.jsonl'
    append_monitor_row(path, {'epoch': 1, 'gradient': float('inf'), 'nested': {'value': float('nan')}, 'new_key': 42})
    row = json.loads(path.read_text())
    assert row == {'epoch': 1, 'gradient': None, 'nested': {'value': None}, 'new_key': 42}
    assert average_diagnostics([{'ratio': 1., 'count': 3}, {'ratio': 2., 'count': 4}]) == {'ratio': 1.5, 'count': 3.5}


def test_legacy_rdi_scale_and_type_bias_receive_module_monitoring():
    model = torch.nn.Module()
    model.backbone = torch.nn.Module()
    model.backbone.dist_bias_scale = torch.nn.Parameter(torch.tensor(1.))
    model.backbone.type_pair_bias = torch.nn.Embedding(9, 1)
    begin_monitor_epoch(model, 1, {})
    (model.backbone.dist_bias_scale + model.backbone.type_pair_bias.weight.sum()).backward()
    snapshot = module_update_snapshot(model)
    assert set(snapshot) == {'rdi'}
    assert len(snapshot['rdi']) == 2
    with torch.no_grad():
        model.backbone.dist_bias_scale.add_(.2)
    finish_module_update(model, snapshot)
    values = plugin_diagnostics(model)
    assert abs(values['module_rdi_grad_norm_unclipped'] - 10**.5) < 1e-6
    assert abs(values['module_rdi_update_norm'] - .2) < 1e-6


def test_input_context_monitor_observes_zero_start_and_first_learned_update():
    from caliroute.plugins.input_encoding import PhysicalInputContextAdapter
    model = torch.nn.Module()
    model.backbone = torch.nn.Module()
    model.backbone.physical_input_adapter = PhysicalInputContextAdapter(8)
    adapter = model.backbone.physical_input_adapter
    begin_monitor_epoch(model, 1, {'monitor_interval': 10})
    output = adapter(torch.ones(2, 3, 12), torch.ones(2, 10))
    output.sum().backward()
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    snapshot = module_update_snapshot(model)
    assert set(snapshot) == {'input_context'}
    optimizer.step()
    finish_module_update(model, snapshot)
    values = plugin_diagnostics(model)
    assert values['input_context_residual_rms'] == 0.
    assert values['module_input_context_grad_norm_unclipped'] > 0.
    assert values['module_input_context_update_norm'] > 0.
    adapter(torch.ones(2, 3, 12), torch.ones(2, 10))
    assert plugin_diagnostics(model)['input_context_residual_rms'] > 0.
    assert all(not tensor.requires_grad for tensor in adapter.diagnostics().values())
    begin_monitor_epoch(model, 2, {'monitor_interval': 10})
    assert not adapter.diagnostics_enabled
    assert plugin_diagnostics(model) == {}
