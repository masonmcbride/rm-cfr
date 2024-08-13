#!/usr/bin/env python3

import sys
import os
import argparse
import json
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from rm_cfr import *

def solve_problem_3_1(game):
    # Constants
    tfsdp1, tfsdp2 = game["decision_problem_pl1"], game["decision_problem_pl2"]
    T = 1_000
    
    # initialize
    p1_cfr = Cfr(tfsdp1)
    y = uniform_sf_strategy(tfsdp2)

    # start the loop
    strategy_sum = {sequence: 0 for sequence in get_sequence_set(tfsdp1)}
    values = []
    for t in range(1,T+1):
        if t % 100 == 0: 
            print(f"doing time {t}") 
        x = p1_cfr.next_strategy()

        utility_1 = compute_utility_vector_pl1(game, y)

        p1_cfr.observe_utility(utility_1)

        strategy_sum = {sequence: current_sum + x[sequence] for sequence,current_sum in strategy_sum.items()}

        average_strategy = {sequence: strategy/t for sequence, strategy in strategy_sum.items()}

        xAy = expected_utility_pl1(game, average_strategy, y)
        values.append(xAy)
    print(f'Value: {values[-1]}')

    fig = plt.figure()
    fig.suptitle(f'Player 1 expected value for {args.game[:-5].upper()} for {T=}\nValue: {values[-1]}', fontsize=10)
    plt.plot(range(T), values, color='r',linewidth=0.5)
    plt.xlabel('T', fontsize=10)
    plt.ylabel('Player 1 EV', fontsize=10)

    plt.savefig(f'hw_plots/prob3.1_{args.game[:-5]}.png')
    plt.show()

def solve_problem_3_2(game):
    # Constants
    tfsdp1, tfsdp2 = game["decision_problem_pl1"], game["decision_problem_pl2"]
    T = 10_000
    
    # initialize
    p1_cfr = Cfr(tfsdp1)
    p2_cfr = Cfr(tfsdp2)

    # start the loop
    p1_cum_strat = {sequence: 0 for sequence in get_sequence_set(tfsdp1)}
    p2_cum_strat = {sequence: 0 for sequence in get_sequence_set(tfsdp2)}
    p1_evs = []
    saddle_points = []
    for t in range(1,T+1):
        if t % 100 == 0: 
            print(f"doing time {t}") 
        x = p1_cfr.next_strategy()
        y = p2_cfr.next_strategy()

        utility_1 = compute_utility_vector_pl1(game, y)
        utility_2 = compute_utility_vector_pl2(game, x)

        p1_cfr.observe_utility(utility_1)
        p2_cfr.observe_utility(utility_2)

        p1_cum_strat = {sequence: current_sum + x[sequence] for sequence,current_sum in p1_cum_strat.items()}
        p2_cum_strat = {sequence: current_sum + y[sequence] for sequence,current_sum in p2_cum_strat.items()}

        p1_avg_strategy = {sequence: strategy/t for sequence, strategy in p1_cum_strat.items()}
        p2_avg_strategy = {sequence: strategy/t for sequence, strategy in p2_cum_strat.items()}

        xAy = expected_utility_pl1(game, p1_avg_strategy, y)
        p1_evs.append(xAy)

        saddle_point_gap = gap(game, p1_avg_strategy, p2_avg_strategy)
        saddle_points.append(saddle_point_gap)
    print(f'Value: {p1_evs[-1]}')
    print(f'Saddle Point Gap: {saddle_points[-1]}')

    fig = plt.figure()
    fig.suptitle(f'Player 1 EV and Saddle Point Gap for {args.game[:-5].upper()} for {T=}\nValue: {p1_evs[-1]} SPG: {saddle_points[-1]}', fontsize=10)
    plt.plot(range(T), p1_evs, color='b',label='player 1 expected value',linewidth=1.0)
    plt.plot(range(T), saddle_points,label='saddle point gap between avg strategies', color='r',linewidth=1.0)
    plt.legend()
    plt.xlabel('Time (t)', fontsize=10)
    plt.ylabel('Value', fontsize=10)

    plt.savefig(f'prob3.2_{args.game[:-5]}.png')
    plt.show()


def solve_problem_3_3(game):
    # Constants
    tfsdp1, tfsdp2 = game["decision_problem_pl1"], game["decision_problem_pl2"]
    T = 10_000
    
    # initialize
    p1_cfr = Cfr_plus(tfsdp1)
    p2_cfr = Cfr_plus(tfsdp2)

    # start the loop
    p1_cum_strat = {sequence: 0 for sequence in get_sequence_set(tfsdp1)}
    p2_cum_strat = {sequence: 0 for sequence in get_sequence_set(tfsdp2)}
    p1_evs = []
    saddle_points = []

    x = p1_cfr.next_strategy()
    for t in range(1,T+1):
        if t % 100 == 0: 
            print(f"doing time {t}") 
        y = p2_cfr.next_strategy()
        utility_1 = compute_utility_vector_pl1(game, y)

        p1_cfr.observe_utility(utility_1)

        x = p1_cfr.next_strategy()
        utility_2 = compute_utility_vector_pl2(game, x)

        p2_cfr.observe_utility(utility_2)

        p1_cum_strat = {sequence: current_sum + x[sequence] for sequence,current_sum in p1_cum_strat.items()}
        p2_cum_strat = {sequence: current_sum + y[sequence] for sequence,current_sum in p2_cum_strat.items()}

        p1_avg_strategy = {sequence: strategy/t for sequence, strategy in p1_cum_strat.items()}
        p2_avg_strategy = {sequence: strategy/t for sequence, strategy in p2_cum_strat.items()}

        xAy = expected_utility_pl1(game, p1_avg_strategy, y)
        p1_evs.append(xAy)

        saddle_point_gap = gap(game, p1_avg_strategy, p2_avg_strategy)
        saddle_points.append(saddle_point_gap)
    print(f'Value: {p1_evs[-1]}')
    print(f'Saddle Point Gap: {saddle_points[-1]}')

    fig = plt.figure()
    fig.suptitle(f'Player 1 EV and Saddle Point Gap with CFR+ and Modified Self Play for {args.game[:-5].upper()} for {T=}\nValue: {p1_evs[-1]} SPG: {saddle_points[-1]}', fontsize=7)
    plt.plot(range(T), p1_evs, color='b',label='player 1 expected value',linewidth=1.0)
    plt.plot(range(T), saddle_points,label='saddle point gap between avg strategies', color='r',linewidth=1.0)
    plt.legend()
    plt.xlabel('Time (t)', fontsize=10)
    plt.ylabel('Value', fontsize=10)

    plt.savefig(f'prob3.3_{args.game[:-5]}.png')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Problem 3 (CFR)')
    parser.add_argument("--game", help="Path to game file")
    parser.add_argument("--problem", choices=["3.1", "3.2", "3.3"])

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
        del node
    del tfsdp
    for entry in game["utility_pl1"]:
        assert isinstance(entry["sequence_pl1"], list)
        assert isinstance(entry["sequence_pl2"], list)
        entry["sequence_pl1"] = tuple(entry["sequence_pl1"])
        entry["sequence_pl2"] = tuple(entry["sequence_pl2"])
    del entry

    print("... done. Running code for Problem", args.problem)

    if args.problem == "3.1":
        solve_problem_3_1(game)
    elif args.problem == "3.2":
        solve_problem_3_2(game)
    else:
        assert args.problem == "3.3"
        solve_problem_3_3(game)
    del game
#pprint(locals())