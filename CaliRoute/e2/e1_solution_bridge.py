"""Execute unchanged E1 solution-reference functions without its actor imports.

Only the explicit host teacher-forcing callback is replaced. Function bodies
and transitive pure helpers are compiled from the frozen trainer.py, retaining
the exact E1 gates, group/reference advantages and cost-incumbent contract.
"""
import ast
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import numpy as np
import torch
from offline2online.slppo_diagnostics import numpy_distribution


def load_solution_functions(teacher):
    path=Path(__file__).resolve().parents[1]/'offline2online/trainer.py'
    source=path.read_text();tree=ast.parse(source)
    definitions={x.name:x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.ClassDef))}
    roots=['_solution_level_advantage_tensors','_prepare_sl_expert_candidates',
           '_apply_solution_level_aliases','_validate_e1_solution_contract']
    selected=set();pending=list(roots)
    while pending:
        name=pending.pop()
        if name in selected or name=='_expert_route_mean_logprobs':continue
        selected.add(name)
        pending.extend(x.id for x in ast.walk(definitions[name])
            if isinstance(x,ast.Name) and x.id in definitions and x.id not in selected)
    nodes=[x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name in selected]
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])
    namespace=dict(np=np,torch=torch,math=math,dataclass=dataclass,numpy_distribution=numpy_distribution,
        E1_SOLUTION_REFERENCE_CONTRACT='e1_cost_incumbent_v1',_expert_route_mean_logprobs=teacher)
    exec(compile(ast.fix_missing_locations(module),str(path),'exec'),namespace)
    return namespace,dict(source=str(path),sha256=hashlib.sha256(source.encode()).hexdigest(),
        unchanged_functions=sorted(selected),host_replaced_functions=['_expert_route_mean_logprobs'])
