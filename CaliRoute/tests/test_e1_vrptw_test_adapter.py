import importlib.util
from pathlib import Path
import pytest
spec = importlib.util.spec_from_file_location('vrptw_test',Path(__file__).parents[1]/'scripts/run_e1_vrptw_test.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def test_test_split_overlap_and_duplicate_ids_rejected():
    train = {'train':{'instances':[{'instance_id':'train','semantic_sha256':'shared'}]},'val':{'instances':[]}}
    test = {'count':1000,'instances':[{'instance_id':str(i),'semantic_sha256':str(i)} for i in range(1000)]}
    m.check_split(train,test)
    test['instances'][0]['semantic_sha256']='shared'
    with pytest.raises(ValueError,match='overlaps'):m.check_split(train,test)
    test['instances'][0]=test['instances'][1]
    with pytest.raises(ValueError,match='Duplicate'):m.check_split(train,test)

def test_candidate_budget_and_checkpoint_contract():
    audit={'instances':[{'instance_id':str(i)} for i in range(1000)]}
    rows=[dict(instance_id=str(i),method='ppo_base',training_seed=3009,split='test',checkpoint_id='sha',actual_K=50,requested_K=50) for i in range(1000)]
    m.check_rows(rows,audit,'ppo_base',3009,'sha')
    rows[0]['actual_K']=49
    with pytest.raises(ValueError,match='mismatch'):m.check_rows(rows,audit,'ppo_base',3009,'sha')
    rows[0]['actual_K']=50;rows[0]['checkpoint_id']='wrong'
    with pytest.raises(ValueError,match='mismatch'):m.check_rows(rows,audit,'ppo_base',3009,'sha')
