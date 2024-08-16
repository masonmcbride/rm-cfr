#!/usr/bin/env python3

import sys
import os
import argparse
import json
import gurobipy as gurobi
from gurobipy import GRB

from pprint import pprint
def LP_realization_matrices(tfsdp) -> tuple:
    """Construct the realization matrices needed for LP solving
    F: The reach_probaility and total constraints probality due to tree structure
    f: Constraint that assures that root reach probability is 1.  
    """
    J = [node for node in tfsdp if node['type'] == 'decision'] 
    sequence_list = [(n['id'],a) for n in J for a in n['actions']]

    # initialize the root realization row
    F = [{**{sequence: 0 for sequence in sequence_list}, None: 1}]
    
    # build the realization matrix for each infoset
    for infoset in J:
        F.append({
            **{sequence: 0 for sequence in sequence_list},
            None: 0,
            infoset['parent_sequence']: -1,
            **{(infoset['id'],a): 1 for a in infoset['actions']}
        })

    f = [1] + [0] * len(J)

    return (F, f)

def to_ndarray(tfsdp, F: list[dict]):
    J = [node for node in tfsdp if node['type'] == 'decision'] 
    sequence_list = [(n['id'],a) for n in J for a in n['actions']]
    sequence_map = dict([(None,0)]+list(zip(sequence_list,range(1,len(sequence_list)+1))))
    for row in F:
        out = [0] * len(row)
        for sequence,val in row.items():
            out[sequence_map[sequence]] = val
        print(out)

def solve_problem_2_1(game):
    tfsdp1, tfsdp2 = game['decision_problem_pl1'], game['decision_problem_pl2']
    tfsdp = {1: tfsdp1, 2: tfsdp2}
    F1, f1 = LP_realization_matrices(tfsdp1)
    F2, f2 = LP_realization_matrices(tfsdp2)
    pprint(F1)
    pprint(F2)

    # first output what the F and f matrix might look like 
    for player in [1, 2]:
        pass


def solve_problem_2_2(game):
    for player in [1, 2]:
        m = gurobi.Model("deterministic_pl" + str(player))

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