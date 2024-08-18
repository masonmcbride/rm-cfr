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
    A = [] 
    for leaf_node in game['utility_pl1']:
        row = rows[leaf_node['sequence_pl2']]
        row[leaf_node['sequence_pl1']] = leaf_node['value']
    for seq in sequence_list_pl2:
        print(seq)
        print(rows[seq].values())
    A = [list(row.values()) for row in sequence_list_pl2]
    pprint(f"{A}")
    return rows

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
    A = build_payoff_matrix(game)
    tfsdp1, tfsdp2 = game['decision_problem_pl1'], game['decision_problem_pl2']
    tfsdp = {1: tfsdp1, 2: tfsdp2}
    F1_dict, f1_dict = LP_realization_matrices(tfsdp1)
    F2_dict, f2_dict = LP_realization_matrices(tfsdp2)
    F1, f1 = to_ndarrays(tfsdp1,F1_dict,f1_dict)
    F2, f2 = to_ndarrays(tfsdp2,F2_dict,f2_dict)
    print(f"{F1.shape=}")
    print(f"{f1.shape=}")
    print(f"{F2.shape=}")
    print(f"{f2.shape=}")

    # first output what the F and f matrix might look like 
    for player in [1, 2]:
        J = [node for node in tfsdp[player] if node['type'] == 'decision'] 

        # initialize model
        m = gurobi.Model(f"Nash Equilibrium for {player=}")

        # define variables
        x = m.addMVar(shape=len(LP_sequence_set(tfsdp[player])), lb=0.0, name='x')
        print(f"{x.shape=}")
        v = m.addMVar(shape=len(J)+1, name='v')
        print(f"{v.shape=}")
        print(f"state before opt {m.status}")

        # set objective
        #m.setObjective(f2@v)

        # add constraints

        # optimize and output
        #m.optimize()
        print(f"status now {m.status}")
        if m.status == GRB.OPTIMAL:
            print("Optimal objective value:", m.objVal)
            print("Optimal strategy x:", x.X)
            print("Optimal variable v:", v.X)
        else:
            print("Optimization was not successful.")
        
        # flip the payoff matrix for player 2
        A = -A.T




def solve_problem_2_2(game):
    for player in [1, 2]:
        m = gurobi.Model(f"deterministic_pl{player}")

        # FINISH
        #
        # To debug your implementation, you might find it useful to ask
        # Gurobi to output the current model it thinks you are asking it
        # to optimize.
        raise NotImplementedError

        m.optimize()

def solve_problem_2_3(game):
    for player in [1, 2]:
        # FINISH
        #
        # To debug your implementation, you might find it useful to ask
        # Gurobi to output the current model it thinks you are asking it
        # to optimize.
        raise NotImplementedError

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