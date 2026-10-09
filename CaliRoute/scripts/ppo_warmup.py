"""Epoch-based pure PPO -> SL-PPO schedule shared by current/archive adapters.

This helper has no project imports so loading it cannot mix archived modules
with current model/environment implementations.
"""
from __future__ import annotations

SL_METHODS = frozenset({'sl_ppo', 'sl-ppo', 'solution_level_ppo',
                        'solution-level-ppo', 'solution_ppo', 'solution-ppo'})
PHASE_FIELDS = ('training_phase', 'effective_offline_method', 'ppo_warmup_epochs',
                'phase_epoch', 'sl_enabled_by_schedule')


class PPOWarmupSchedule:
    def __init__(self, cfg):
        training = cfg.get('training', {}) or {}
        offline = cfg.get('offline', {}) or {}
        value = training.get('ppo_warmup_epochs', 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError('training.ppo_warmup_epochs must be a nonnegative integer')
        self.epochs = value
        self.method = str(offline.get('method', 'ppo')).strip().lower()
        if self.epochs:
            if self.method not in SL_METHODS:
                raise ValueError('PPO warmup requires a configured SL-PPO method after warmup')
            if (offline.get('use_priority_sampler', False)
                    or training.get('use_oracle_ordering_hint', offline.get('use_oracle_ordering_hint', False))
                    or int(offline.get('bc_warmup_epochs', 0) or 0) > 0
                    or float(offline.get('hard_ref_kl_coef', offline.get('lambda_ref_kl', 0.)) or 0.) > 0.):
                raise ValueError('Pure PPO warmup cannot use expert priority/oracle/BC warmup/reference KL')

    def warming(self, epoch):
        return self.epochs > 0 and 1 <= int(epoch) <= self.epochs

    def fields(self, epoch):
        epoch = int(epoch)
        warming = self.warming(epoch)
        return dict(training_phase='ppo_warmup' if warming else ('slppo' if self.method in SL_METHODS else self.method),
                    effective_offline_method='ppo' if warming else self.method,
                    ppo_warmup_epochs=self.epochs,
                    phase_epoch=epoch if warming or not self.epochs else max(epoch-self.epochs, 0),
                    sl_enabled_by_schedule=not warming and self.method in SL_METHODS)

    def checkpoint_metadata(self, epoch):
        return dict(self.fields(epoch), completed_epoch=int(epoch),
                    next_effective_offline_method=self.fields(int(epoch)+1)['effective_offline_method'],
                    transition='same_model_optimizer_scaler_and_rng; no_checkpoint_reload_or_reset',
                    warmup_boundary_checkpoint=int(epoch) == self.epochs and self.epochs > 0,
                    checkpoint_selection='best_validation_over_all_trained_epochs; epoch_zero_excluded')


class OriginalPPOWarmupRuntime:
    """External epoch gates; archived files and loss implementations stay intact.

    f388343 caches its configured method once. Gate its SL predicate at the
    existing epoch-start callback and bypass auxiliary advantages during warmup.
    The original PPO update function and every optimizer boundary are unchanged.
    """
    _AUX_COEFFICIENTS = ('sl_coef', 'notclose_coef', 'premature_close_coef',
                         'member_coef', 'onpolicy_member_coef',
                         'anchor_coef', 'onpolicy_anchor_coef')

    def __init__(self, schedule, *, is_primary=True):
        self.schedule = schedule
        self.is_primary = is_primary
        self.epoch = 0
        self.originals = {}
        self.runtime_cfg = None
        self.saved_coefficients = None
        self.optimizer = self.agent = None
        self.boundary_saved = False

    def _restore_coefficients(self, offline):
        for key in self._AUX_COEFFICIENTS:
            if key in self.saved_coefficients:
                offline[key] = self.saved_coefficients[key]
            else:
                offline.pop(key, None)

    def install(self, trainer):
        if not self.schedule.epochs:
            return
        import copy
        import json
        from pathlib import Path
        runtime = self

        def patch(name, wrapper):
            self.originals[name] = getattr(trainer, name)
            setattr(trainer, name, wrapper)

        original_scale = trainer.pbrs_scale_for_epoch
        original_predicate = trainer._is_solution_level_method
        original_aux = trainer._apply_auxiliary_advantages
        original_step = trainer._optimizer_step
        original_save = trainer.save_checkpoint

        def scale(cfg, epoch, epochs):
            runtime.epoch = int(epoch)
            runtime.runtime_cfg = cfg
            offline = cfg.setdefault('offline', {})
            if runtime.saved_coefficients is None:
                runtime.saved_coefficients = {k: offline[k] for k in runtime._AUX_COEFFICIENTS if k in offline}
            runtime._restore_coefficients(offline)
            if runtime.schedule.warming(epoch):
                offline.update({key: 0. for key in runtime._AUX_COEFFICIENTS})
            if epoch in (1, runtime.schedule.epochs+1):
                print('[TrainingPhase] ' + json.dumps(dict(epoch=epoch, **runtime.schedule.fields(epoch)), sort_keys=True), flush=True)
            return original_scale(cfg, epoch, epochs)

        def predicate(method):
            return not runtime.schedule.warming(runtime.epoch) and original_predicate(method)

        def auxiliary(advantages, *args, **kwargs):
            if runtime.schedule.warming(runtime.epoch):
                return advantages, {}
            return original_aux(advantages, *args, **kwargs)

        def step(optimizer, agent, *args, **kwargs):
            runtime.optimizer, runtime.agent = optimizer, agent
            return original_step(optimizer, agent, *args, **kwargs)

        def save(path, agent, optimizer, cfg, epoch, seed):
            saved_cfg = copy.deepcopy(cfg)
            if runtime.saved_coefficients is not None:
                runtime._restore_coefficients(saved_cfg.setdefault('offline', {}))
            saved_cfg['training_stage'] = runtime.schedule.checkpoint_metadata(epoch)
            result = original_save(path, agent, optimizer, saved_cfg, epoch, seed)
            if runtime.is_primary:
                sidecar = Path(path).with_suffix('.phase.json')
                temp = sidecar.with_name('.' + sidecar.name + '.tmp')
                temp.write_text(json.dumps(runtime.schedule.checkpoint_metadata(epoch), indent=2) + '\n')
                temp.replace(sidecar)
            return result

        for name, wrapper in (('pbrs_scale_for_epoch', scale), ('_is_solution_level_method', predicate),
                              ('_apply_auxiliary_advantages', auxiliary), ('_optimizer_step', step),
                              ('save_checkpoint', save)):
            patch(name, wrapper)

    def finish_epoch(self, trainer, row, seed):
        """Guarantee the actual boundary checkpoint, even off the save interval."""
        if not self.schedule.epochs:
            return row
        from pathlib import Path
        epoch = int(row['epoch'])
        row = dict(row, **self.schedule.fields(epoch))
        row['train_mode'] = row['effective_offline_method']
        if epoch == self.schedule.epochs and not self.boundary_saved:
            if self.optimizer is None or self.agent is None:
                raise RuntimeError('PPO warmup boundary reached without an optimizer step')
            cfg = self.runtime_cfg
            data = cfg['data']
            problem = trainer.problem_type_from_config(cfg)
            stations = trainer.num_charging_stations_for_problem(data, problem, evrptw_default=10)
            path = (Path(trainer.REPO_ROOT) / 'results/checkpoints'
                    / f"Cus_{int(data['num_customers'])}_CS_{stations}" / cfg['run_name']
                    / f'seed_{seed}' / f'checkpoint_epoch_{epoch:04d}.pt')
            trainer.save_checkpoint(path, self.agent, self.optimizer, cfg, epoch, seed)
            self.boundary_saved = True
        return row

    def uninstall(self, trainer):
        if self.runtime_cfg is not None and self.saved_coefficients is not None:
            self._restore_coefficients(self.runtime_cfg['offline'])
        for name, function in self.originals.items():
            setattr(trainer, name, function)
        self.originals.clear()
