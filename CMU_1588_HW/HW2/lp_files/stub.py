#!/usr/bin/env python3

import os
import argparse
import json
import gurobipy as gurobi
from gurobipy import GRB


def solve_problem_2_1(game):
    A = build_payoff_matrix()
    F1, F2, f1, f2 = build_realization_weights()

    for player in [1, 2]:

        # create model
        m = gurobi.Model("game_value_pl" + str(player))
        
        # create variables
        x = m.addMVar(3, lb=None, up=1.0, name="strategy_vec")
        v = m.addMVar(1, name="value_vec")

        # create objective
        m.setObjective(v, GRB.MAXIMIZE)

        # add constraints
        m.addConstr(A @ x - F1 @ v >= 0)
        m.addConstr(F1 @ x == f1)
        # To debug your implementation, you might find it useful to ask
        # Gurobi to output the current model it thinks you are asking it
        # to optimize.

        m.optimize()


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
