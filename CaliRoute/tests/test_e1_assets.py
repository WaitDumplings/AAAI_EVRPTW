from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import pytest

from e1.assets import audit_assets
from e1.validator import validate_routes,recover_routes_from_arcs,routes_from_sequence,teacher_force_routes
from offline2online.instance_adapter import adapt_instance_payload


def payload(identity='train_0',factor=1.):
    return dict(instance_id=identity,problem_class='CVRP',depot=np.array([100.,100.]),
        customers=np.array([[999.,999.],[200.,100.]]),
        distance_matrix_km=np.array([[0.,1.,5.],[4.,0.,2.],[6.,3.,0.]],dtype=np.float32)*factor,
        demands_cm3=np.array([1.,1.],dtype=np.float32),vehicle={'cargo_capacity_cm3':2.},
        metadata={'dataset_version':'fixture_v1','service_territory_id':'one','objective_unit':'km'})


def instance():return adapt_instance_payload(payload(),problem_type='cvrp',strict_road_metric=True)


def write_csv(path,rows):
    fields=sorted(set().union(*(row.keys() for row in rows))) if rows else ['instance_id','objective_distance_km','routes_json']
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def fixture(tmp_path,rows=None,train=None):
    root=tmp_path/'AAAI_Dataset'
    train=train or payload()
    for split,item in [('train',train),('val',payload('val_0',1.1)),('test',payload('test_0',1.2))]:
        path=root/('test_release' if split=='test' else 'dataset')/'cvrp'/split/'Cus2';path.mkdir(parents=True)
        with(path/'instances.pkl').open('wb')as f:pickle.dump({'instances':[item]},f)
        (path/'metadata.json').write_text(json.dumps(dict(num_instances=1,dataset_version='fixture_v1',
            bundle_sha256=hashlib.sha256((path/'instances.pkl').read_bytes()).hexdigest())))
    if rows is None:rows=[dict(instance_id='train_0',objective_distance_km=9.,routes_json='[[0,1,2,0]]')]
    for name in ['expert_solutions.csv','gurobi_summary.csv']:write_csv(root/'dataset/cvrp/train/Cus2'/name,rows)
    return root


def test_independent_validator_uses_asymmetric_matrix_and_return_edges():
    inst=instance();result=validate_routes(inst,[[0,1,2,0]],9.)
    assert result['valid'] and result['recomputed_distance_km']==9.
    assert validate_routes(inst,[[0,2,1,0]],12.)['valid']
    assert not validate_routes(inst,[[0,1,2]],3.)['valid']
    assert not validate_routes(inst,[[0,1,1,2,0]],10.)['valid']
    assert not validate_routes(inst,[[0,True,2,0]],9.)['valid']
    assert not validate_routes(inst,[[0,1.,2,0]],9.)['valid']


def test_capacity_and_nonfinite_edges_rejected():
    inst=instance();inst.vehicle['cargo_capacity_cm3']=1.
    assert 'capacity_exceeded' in validate_routes(inst,[[0,1,2,0]],9.)['reasons']
    inst.distance_matrix_km[1,2]=np.inf
    assert 'invalid_road_edge' in validate_routes(inst,[[0,1,2,0]],np.inf)['reasons']


@pytest.mark.parametrize('routes',[[None],[1],['bad'],[{},[0,1,2,0]]])
def test_malformed_candidate_route_is_reported_not_raised(routes):
    result=validate_routes(instance(),routes)
    assert not result['feasible']
    assert 'invalid_route_shape' in result['reasons']


def test_valid_expert_teacher_forces_in_cvrp_env():
    result=teacher_force_routes(instance(),[[0,1,2,0]])
    assert result['valid'] and result['completed'] and not result['truncated']
    assert result['recomputed_cost_km']==9.
    bad=teacher_force_routes(instance(),[[0,1,1,2,0]])
    assert not bad['valid'] and bad['reason']=='expert_action_mask_violation'


def test_audit_builds_one_shared_train_only_pool_and_rejects_test_expert_id(tmp_path):
    rows=[dict(instance_id='train_0',objective_distance_km=9.,routes_json='[[0,1,2,0]]'),
          dict(instance_id='test_0',objective_distance_km=10.8,routes_json='[[0,1,2,0]]')]
    root=fixture(tmp_path,rows);report=audit_assets(root,tmp_path/'audit',expected_customers=2)
    assert report['training_data_ready'] and report['expert_methods_ready'] and report['test_data_ready']
    pool=report['expert_pool'];assert pool['accepted_count']==1 and pool['coverage']==1.
    assert pool['failure_reasons']['expert_id_not_in_train']==1
    assert pool['counts']['teacher_forcing_feasible']==pool['counts']['recorded_cost_consistent']==1
    with Path(pool['path']).open()as f:accepted=list(csv.DictReader(f))
    assert [r['instance_id'] for r in accepted]==['train_0']
    assert accepted[0]['source_split']=='train'
    assert all(s['territory_counts']=={'one':1} for s in report['splits'].values())
    assert all(v['semantic_sha256']['count']==0 for v in report['split_overlaps'].values())
    assert report['splits']['test']['path'].endswith('test_release/cvrp/test/Cus2')


def test_cost_only_never_invents_route_and_ppo_assets_remain_ready(tmp_path):
    root=fixture(tmp_path,[dict(instance_id='train_0',objective_distance_km=9.)])
    report=audit_assets(root,tmp_path/'audit',expected_customers=2)
    assert report['training_data_ready'] and not report['expert_methods_ready']
    assert report['expert_pool']['counts']['with_recorded_incumbent']==1
    assert report['expert_pool']['accepted_count']==0


def test_deterministic_arc_recovery_and_missing_cost_recomputation(tmp_path):
    arcs={'x[2,0]':1,'x[0,1]':1,'x[1,2]':1,'x[0,2]':0}
    assert recover_routes_from_arcs(arcs)==[[0,1,2,0]]
    root=fixture(tmp_path,[dict(instance_id='train_0',arc_variables_json=json.dumps(arcs))])
    report=audit_assets(root,tmp_path/'audit',expected_customers=2)
    assert report['expert_pool']['accepted_count']==1
    assert report['expert_pool']['counts']['cost_recovered_from_complete_route']==1
    assert report['expert_pool']['counts']['with_recorded_incumbent']==0


@pytest.mark.parametrize('arcs',[[[0,1],[1,0],[2,3],[3,2]],[[0,1],[0,2],[1,2],[2,0]],[[0,1,.5],[1,0,1]]])
def test_ambiguous_or_disconnected_arcs_are_not_solved_for_user(arcs):
    with pytest.raises(ValueError):recover_routes_from_arcs(arcs)


def test_missing_depot_not_silently_repaired():
    with pytest.raises(ValueError):routes_from_sequence([0,1,2])
    assert routes_from_sequence([0,1,0,2,0])==[[0,1,0],[0,2,0]]


def test_mismatched_cost_rejected_not_silently_relabelled(tmp_path):
    root=fixture(tmp_path,[dict(instance_id='train_0',objective_distance_km=1.,routes_json='[[0,1,2,0]]')])
    report=audit_assets(root,tmp_path/'audit',expected_customers=2)
    assert report['expert_pool']['accepted_count']==0
    assert report['expert_pool']['failure_reasons']['recorded_cost_mismatch']==1


def test_skipping_teacher_forcing_cannot_produce_train_ready_experts(tmp_path):
    root=fixture(tmp_path);report=audit_assets(root,tmp_path/'audit',expected_customers=2,teacher_force=False)
    assert report['training_data_ready'] and not report['expert_methods_ready']
    assert report['expert_pool']['accepted_count']==0


def test_overlap_by_physical_content_blocks_pool_even_when_ids_differ(tmp_path):
    train=payload('train_0',1.1);root=fixture(tmp_path,train=train)
    report=audit_assets(root,tmp_path/'audit',expected_customers=2)
    assert not report['training_data_ready']
    assert report['split_overlaps']['train:val']['semantic_sha256']['count']==1
    assert report['expert_pool']['accepted_count']==0


def test_missing_expert_files_do_not_block_data_only_methods(tmp_path):
    root=fixture(tmp_path)
    for name in ['expert_solutions.csv','gurobi_summary.csv']:(root/'dataset/cvrp/train/Cus2'/name).unlink()
    report=audit_assets(root,tmp_path/'audit',expected_customers=2)
    assert report['training_data_ready'] and not report['expert_methods_ready']
    assert len(report['expert_pool']['missing_assets'])==2


def test_output_never_overwrites_existing_audit(tmp_path):
    root=fixture(tmp_path);out=tmp_path/'existing';out.mkdir();sentinel=out/'keep';sentinel.write_text('safe')
    with pytest.raises(FileExistsError):audit_assets(root,out,expected_customers=2)
    assert sentinel.read_text()=='safe'
