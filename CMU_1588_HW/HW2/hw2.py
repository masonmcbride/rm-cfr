#!/usr/bin/env python3

import sys
import os
import argparse
import json
import gurobipy as gurobi
from gurobipy import GRB

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from rm_cfr import *
import numpy as np

from pprint import pprint

def build_payoff_matrix(game):
    """contruct the payoff matrix from the json provided in the game variable
    turn the sequence form payoff entries into payoff matrix wrt player 1
    since it zero sum the payoff matrix for player 2 is -A
    """
    infoset_pl1 = [n for n in game['decision_problem_pl1'] if n['type'] == 'decision'] 
    sequence_list_pl1 = [None] + [(n['id'],a) for n in infoset_pl1 for a in n['actions']]

    infoset_pl2 = [n for n in game['decision_problem_pl2'] if n['type'] == 'decision']
    sequence_list_pl2 = [None] + [(n['id'],a) for n in infoset_pl2 for a in n['actions']]

    rows = {sequence_pl2: {sequence_pl1: 0 for sequence_pl1 in sequence_list_pl1} 
            for sequence_pl2 in sequence_list_pl2} 
    for leaf_node in game['utility_pl1']:
        row = rows[leaf_node['sequence_pl2']]
        row[leaf_node['sequence_pl1']] = leaf_node['value']
    A = [list(rows[seq].values()) for seq in sequence_list_pl2] 
    return np.array(A)

def LP_realization_matrices(tfsdp) -> tuple:
    """Construct the realization matrices needed for LP solving
    F: The reach_probaility and total constraints probality due to tree structure
    f: Constraint that assures that root reach probability is 1.  
    """
    J = [node for node in tfsdp if node['type'] == 'decision'] 
    sequence_list = [(n['id'],a) for n in J for a in n['actions']]

    # initialize the root realization row
    F = [{None: 1, **{sequence: 0 for sequence in sequence_list}}]
    
    # build the realization matrix for each infoset
    for infoset in J:
        F.append({
            **{sequence: 0 for sequence in sequence_list},
            None: 0,
            infoset['parent_sequence']: -1,
            **{(infoset['id'],a): 1 for a in infoset['actions']}
        })

    f = [{None: 1, **{infoset['id']: 0 for infoset in J}}]

    return (F, f)

def LP_sequence_set(tfsdp) -> set:
    """return sequence set with empty sequuence for LP conditions"""
    return get_sequence_set(tfsdp).union({None})

def LP_uniform_strategy(tfsdp) -> dict:
    """return uniform strategy for LP (including the empty sequence with 1 probability)"""
    uniform = uniform_sf_strategy(tfsdp)
    uniform[None] = 1
    return uniform

def to_ndarrays(tfsdp, F_dict: list[dict], f_dict):
    """Turns the dic"""
    infosets = [node for node in tfsdp if node['type'] == 'decision'] 
    sequence_list = [None] + [(n['id'],a) for n in infosets for a in n['actions']]

    F = [[row[sequence] for sequence in sequence_list] for row in F_dict]
    f = [[1] + [0] * len(infosets)]

    return np.array(F), np.array(f)

def solve_problem_2_1(game):
    tfsdp1, tfsdp2 = game['decision_problem_pl1'], game['decision_problem_pl2']
    A = build_payoff_matrix(game)
    F1_dict, f1_dict = LP_realization_matrices(tfsdp1)
    F2_dict, f2_dict = LP_realization_matrices(tfsdp2)
    F1, f1 = to_ndarrays(tfsdp1,F1_dict,f1_dict)
    F2, f2 = to_ndarrays(tfsdp2,F2_dict,f2_dict)
    infosets_pl1 = [node for node in tfsdp1 if node['type'] == 'decision']
    infosets_pl2 = [node for node in tfsdp2 if node['type'] == 'decision']
    print(f"{A.shape=}")
    print(f"{F1.shape=}")
    print(f"{f1.shape=}")
    print(f"{F2.shape=}")
    print(f"{f2.shape=}")

    #### PLayer 1 LP ####
    m1 = gurobi.Model("Nash Equilbrium for player 1")

    # define variables
    x = m1.addMVar(shape=len(LP_sequence_set(tfsdp1)), lb=0.0, ub=1.0, name='x')
    print(f"{x.shape=}")
    v1 = m1.addMVar(shape=len(infosets_pl1)+1, lb=-10, ub=10, name='v1')
    print(f"{v1.shape=}")

    # set objective
    m1.setObjective(f2@v1,GRB.MAXIMIZE)

    # add constraints
    m1.addConstr(A@x - F2.T@v1 >= 0)
    m1.addConstr(F1@x == f1)

    # optimize and output
    m1.optimize()

    #### PLayer 2 LP ####
    m2 = gurobi.Model("Nash Equilbrium for player 2")

    # define variables
    y = m2.addMVar(shape=len(LP_sequence_set(tfsdp2)), lb=0.0, ub=1.0, name='y')
    print(f"{y.shape=}")
    v2 = m2.addMVar(shape=len(infosets_pl2)+1, lb=-10, ub=10, name='v2')
    print(f"{v2.shape=}")

    # set objective
    m2.setObjective(f1@v2,GRB.MAXIMIZE)

    # add constraints
    m2.addConstr(-A.T@y - F1.T@v2 >= 0)
    m2.addConstr(F2@y == f2)

    # optimize and output
    m2.optimize()

    print(f"The sum of both objective values should be 0.\nResult: {m1.objVal + m2.objVal}")

def solve_problem_2_2(game):
    tfsdp1, tfsdp2 = game['decision_problem_pl1'], game['decision_problem_pl2']
    A = build_payoff_matrix(game)
    F1_dict, f1_dict = LP_realization_matrices(tfsdp1)
    F2_dict, f2_dict = LP_realization_matrices(tfsdp2)
    F1, f1 = to_ndarrays(tfsdp1,F1_dict,f1_dict)
    F2, f2 = to_ndarrays(tfsdp2,F2_dict,f2_dict)
    infosets_pl1 = [node for node in tfsdp1 if node['type'] == 'decision']
    infosets_pl2 = [node for node in tfsdp2 if node['type'] == 'decision']
    print(f"{A.shape=}")
    print(f"{F1.shape=}")
    print(f"{f1.shape=}")
    print(f"{F2.shape=}")
    print(f"{f2.shape=}")

    #### PLayer 1 LP ####
    m1 = gurobi.Model("Optimal deterministic strategy for player 1")

    # define variables
    x = m1.addMVar(shape=len(LP_sequence_set(tfsdp1)), vtype=GRB.BINARY, name='x')
    print(f"{x.shape=}")
    v1 = m1.addMVar(shape=len(infosets_pl1)+1, lb=-10, ub=1, name='v1')
    print(f"{v1.shape=}")

    # set objective
    m1.setObjective(f2@v1,GRB.MAXIMIZE)

    # add constraints
    m1.addConstr(A@x - F2.T@v1 >= 0)
    m1.addConstr(F1@x == f1)

    # optimize and output
    m1.optimize()

    #### PLayer 2 LP ####
    m2 = gurobi.Model("Optimal deterministic strategy for player 2")

    # define variables
    y = m2.addMVar(shape=len(LP_sequence_set(tfsdp2)), vtype=GRB.BINARY, name='y')
    print(f"{y.shape=}")
    v2 = m2.addMVar(shape=len(infosets_pl2)+1, lb=-10, ub=1, name='v2')
    print(f"{v2.shape=}")

    # set objective
    m2.setObjective(f1@v2,GRB.MAXIMIZE)

    # add constraints
    m2.addConstr(-A.T@y - F1.T@v2 >= 0)
    m2.addConstr(F2@y == f2)

    # optimize and output
    m2.optimize()

    print(f"The sum of both objective values should be 0.\nResult: {m1.objVal + m2.objVal}")
    print(x.X)
    print(y.X)
    print(f"Player 1 nash value {m1.objVal}")
    print(f"Player 2 nash value {m2.objVal}")

def solve_problem_2_3(game):
    tfsdp1, tfsdp2 = game['decision_problem_pl1'], game['decision_problem_pl2']
    A = build_payoff_matrix(game)
    F1_dict, f1_dict = LP_realization_matrices(tfsdp1)
    F2_dict, f2_dict = LP_realization_matrices(tfsdp2)
    F1, f1 = to_ndarrays(tfsdp1,F1_dict,f1_dict)
    F2, f2 = to_ndarrays(tfsdp2,F2_dict,f2_dict)
    infosets_pl1 = [node for node in tfsdp1 if node['type'] == 'decision']
    infosets_pl2 = [node for node in tfsdp2 if node['type'] == 'decision']
    sequence_list_pl1 = [None] + [(n['id'],a) for n in infosets_pl1 for a in n['actions']]
    sequence_list_pl2 = [None] + [(n['id'],a) for n in infosets_pl2 for a in n['actions']]
    sequence_map_pl1 = dict(zip(sequence_list_pl1,range(len(sequence_list_pl1))))
    sequence_map_pl2 = dict(zip(sequence_list_pl2,range(len(sequence_list_pl2))))
    print(f"{A.shape=}")
    print(f"{F1.shape=}")
    print(f"{f1.shape=}")
    print(f"{F2.shape=}")
    print(f"{f2.shape=}")

    print(f"{len(infosets_pl1)=}")
    k = 2
    #### PLayer 1 LP ####
    m1 = gurobi.Model("Controlling amount of determinism for player 1")

    # define variables
    x = m1.addMVar(shape=len(LP_sequence_set(tfsdp1)), lb=0.0, ub=1.0, name='x')
    print(f"{x.shape=}")
    z1 = m1.addMVar(shape=len(LP_sequence_set(tfsdp1)), vtype=GRB.BINARY, name='z')
    print(f"{z1.shape=}")
    v1 = m1.addMVar(shape=len(infosets_pl1)+1, lb=-10, ub=10, name='v1')
    print(f"{v1.shape=}")

    # set objective
    m1.setObjective(f2@v1,GRB.MAXIMIZE)

    # add constraints
    m1.addConstr(A@x - F2.T@v1 >= 0)
    m1.addConstr(F1@x == f1)
    for row in F1:
        data = {int(val): [i for i,v in enumerate(row) if v == val] for val in set(row)}
        print(data)
        for ja in data[1]:
            match data.get(-1, None):
                case [0]: m1.addConstr(x[ja] >= z1[ja])
                case [pj]: m1.addConstr(x[ja] >= x[pj] + z1[ja] - 1)
                case None: m1.addConstr(x[ja] >= z1[ja])
        m1.addConstr(sum(z1[ja] for ja in data[1]) <= 1)
    m1.addConstr(z1.sum() >= k)
    m1.printStats()

    # optimize and output
    m1.optimize()
    print(x.X)
    print(z1.X)

    """
    #### PLayer 2 LP ####
    m2 = gurobi.Model("Controlling amount of determinism for player 2")

    # define variables
    y = m2.addMVar(shape=len(LP_sequence_set(tfsdp2)), lb=0.0, ub=1.0, name='y')
    print(f"{y.shape=}")
    v2 = m2.addMVar(shape=len(infosets_pl2)+1, lb=-10, ub=10, name='v2')
    print(f"{v2.shape=}")

    # set objective
    m2.setObjective(f1@v2,GRB.MAXIMIZE)

    # add constraints
    m2.addConstr(-A.T@y - F1.T@v2 >= 0)
    m2.addConstr(F2@y == f2)

    # optimize and output
    m2.optimize()

    print(f"The sum of both objective values should be 0.\nResult: {m1.objVal + m2.objVal}")
    """

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='HW2 Problem 2 (Deterministic strategies)')
    parser.add_argument("--game", help="Path to game file", required=True)
    parser.add_argument(
        "--problem", choices=["2.1", "2.2", "2.3"], required=True)

    args = parser.parse_args()
    print("Reading game path %s..." % args.game)

    game = json.load(open(args.game))

    # Convert all sequences from lists to tuples
    for tfsdp in [game["decision_problem_pl1"], game["decision_problem_pl2"]]:
        for node in tfsdp:
            if isinstance(node["parent_edge"], list):
                node["parent_edge"] = tuple(node["parent_edge"])
            if "parent_sequence" in node and isinstance(node["parent_sequence"], list):
                node["parent_sequence"] = tuple(node["parent_sequence"])
    for entry in game["utility_pl1"]:
        assert isinstance(entry["sequence_pl1"], list)
        assert isinstance(entry["sequence_pl2"], list)
        entry["sequence_pl1"] = tuple(entry["sequence_pl1"])
        entry["sequence_pl2"] = tuple(entry["sequence_pl2"])

    print("... done. Running code for Problem", args.problem)

    if args.problem == "2.1":
        solve_problem_2_1(game)
    elif args.problem == "2.2":
        solve_problem_2_2(game)
    else:
        assert args.problem == "2.3"
        solve_problem_2_3(game)