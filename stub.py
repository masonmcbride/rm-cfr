#!/usr/bin/env python3

import os
import argparse
import json
from pprint import pprint
from sre_constants import GROUPREF_EXISTS

###############################################################################
# The next functions are already implemented for your convenience
#
# In all the functions in this stub file, `game` is the parsed input game json
# file, whereas `tfsdp` is either `game["decision_problem_pl1"]` or
# `game["decision_problem_pl2"]`.
#
# See the homework handout for a description of each field.

from typing import Mapping

Map = Mapping
Action, Sequence, RM_ID, Node = str, str, str, str
Regret, prob, val, cumulate = float, float, float, float
Strategy = Map[Sequence, prob]
Utility = Map[Sequence, val]


def get_sequence_set(tfsdp):
    """Returns a set of all sequences in the given tree-form sequential decision
    process (TFSDP)"""

    sequences = set()
    for node in tfsdp:
        if node["type"] == "decision":
            for action in node["actions"]:
                sequences.add((node["id"], action))
    return sequences


def is_valid_RSigma_vector(tfsdp, obj):
    """Checks that the given object is a dictionary keyed on the set of sequences
    of the given tree-form sequential decision process (TFSDP)"""

    sequence_set = get_sequence_set(tfsdp)
    return isinstance(obj, dict) and obj.keys() == sequence_set


def assert_is_valid_sf_strategy(tfsdp, obj):
    """Checks whether the given object `obj` represents a valid sequence-form
    strategy vector for the given tree-form sequential decision process
    (TFSDP)"""

    if not is_valid_RSigma_vector(tfsdp, obj):
        print("The sequence-form strategy should be a dictionary with key set equal to the set of sequences in the game")
        os.exit(1)
    for node in tfsdp:
        if node["type"] == "decision":
            parent_reach = 1.0
            if node["parent_sequence"] is not None:
                parent_reach = obj[node["parent_sequence"]]
            if abs(sum([obj[(node["id"], action)] for action in node["actions"]]) - parent_reach) > 1e-3:
                n_id = node["id"]
                print(f"At node ID {n_id} the sum of the child sequences is not equal to the parent sequence")

def best_response_value(tfsdp, utility):
    """Computes the value of max_{x in Q} x^T utility, where Q is the
    sequence-form polytope for the given tree-form sequential decision
    process (TFSDP)"""
    # utility: Mapping[Sequence, utility]

    assert is_valid_RSigma_vector(tfsdp, utility)

    utility_ = utility.copy()
    del utility
    utility_[None] = 0.0
    for node in tfsdp[::-1]:
        if node["type"] == "decision":
            max_ev = max([utility_[(node["id"], action)] for action in node["actions"]])
            utility_[node["parent_sequence"]] += max_ev
    return utility_[None]

def compute_utility_vector_pl1(game, sf_strategy_pl2):
    """Returns A * y, where A is the payoff matrix of the game and y is
    the given strategy for Player 2"""

    assert_is_valid_sf_strategy(game["decision_problem_pl2"], sf_strategy_pl2)

    sequence_set = get_sequence_set(game["decision_problem_pl1"])
    utility = {sequence: 0.0 for sequence in sequence_set}
    for entry in game["utility_pl1"]:
        utility[entry["sequence_pl1"]] += entry["value"] * sf_strategy_pl2[entry["sequence_pl2"]]

    assert is_valid_RSigma_vector(game["decision_problem_pl1"], utility)
    return utility


def compute_utility_vector_pl2(game, sf_strategy_pl1):
    """Returns -A^transpose * x, where A is the payoff matrix of the
    game and x is the given strategy for Player 1"""

    assert_is_valid_sf_strategy(
        game["decision_problem_pl1"], sf_strategy_pl1)

    sequence_set = get_sequence_set(game["decision_problem_pl2"])
    utility = {sequence: 0.0 for sequence in sequence_set}
    for entry in game["utility_pl1"]:
        utility[entry["sequence_pl2"]] -= entry["value"] * sf_strategy_pl1[entry["sequence_pl1"]]

    assert is_valid_RSigma_vector(game["decision_problem_pl2"], utility)
    return utility


def gap(game, sf_strategy_pl1, sf_strategy_pl2):
    """Computes the saddle point gap of the given sequence-form strategies
    for the players"""

    assert_is_valid_sf_strategy(game["decision_problem_pl1"], sf_strategy_pl1)
    assert_is_valid_sf_strategy(game["decision_problem_pl2"], sf_strategy_pl2)

    utility_pl1 = compute_utility_vector_pl1(game, sf_strategy_pl2)
    utility_pl2 = compute_utility_vector_pl2(game, sf_strategy_pl1)

    return (best_response_value(game["decision_problem_pl1"], utility_pl1)
            + best_response_value(game["decision_problem_pl2"], utility_pl2))


###########################################################################
# Starting from here, you should fill in the implementation of the
# different functions

def succ(tfsdp, v, a):
    # O(N)
    # solution: precompute the mapping that precomputes the parent_edge to the node["id"]
    # parent_edge is needed but parent_sequence can be computed when you need
    # most algorithms need the parent_sequence
    for node in tfsdp:
        if node["parent_edge"] == (v, a):
            return node["id"]
    return None

def expected_utility_pl1(game, sf_strategy_pl1, sf_strategy_pl2) -> val:
    """Returns the expected utility for Player 1 in the game, when the two
    players play according to the given strategies"""

    assert_is_valid_sf_strategy(game["decision_problem_pl1"], sf_strategy_pl1)
    assert_is_valid_sf_strategy(game["decision_problem_pl2"], sf_strategy_pl2)

    Ay: Map[Sequence, val] = compute_utility_vector_pl1(game, sf_strategy_pl2)
    xAy: val = sum([sf_strategy_pl1[sequence]*Ay[sequence] for sequence in sf_strategy_pl1])
    return xAy


def uniform_sf_strategy(tfsdp):
    """Returns the uniform sequence-form strategy for the given tree-form
    sequential decision process"""

    # FINISH
    uniform_strategy: Map[Sequence, prob] = {}
    for node in tfsdp:
        if node["type"] == "decision":
            parent_reach = 1.0
            if node["parent_sequence"] is not None:
                parent_reach = uniform_strategy[node["parent_sequence"]]
            for action in node["actions"]:
                uniform_strategy[(node["id"], action)] = parent_reach/len(node["actions"])

    assert_is_valid_sf_strategy(tfsdp, uniform_strategy)
    return uniform_strategy


class RegretMatching(object):
    def __init__(self, action_set):
        self.action_set = set(action_set)
        self.regret_sum: Map[Action, cumulate] = {action: 0 for action in self.action_set}

    def next_strategy(self) -> Map[Action, prob]:
        r_plus: Map[Action, Regret] = {action: max(regret, 0) for action, regret in self.regret_sum.items()}
        if not any(r_plus.values()):
            return {action: 1./len(self.action_set) for action in self.action_set}
        else:
            normalizer = sum(r_plus.values())
            return {action: regret/normalizer for action, regret in r_plus.items()}

    def observe_utility(self, utility: Utility):
        assert isinstance(utility, dict) and utility.keys() == self.action_set

        x: Map[Action, prob] = self.next_strategy()
        ground_truth: val = sum(x[action] * utility[action] for action in self.action_set)

        for action in self.action_set:
            self.regret_sum[action] += utility[action] - ground_truth

"""
class RegretMatchingPlus(object):
    def __init__(self, action_set):
        self.action_set = set(action_set)

        # FINISH
        raise NotImplementedError

    def next_strategy(self):
        # FINISH
        # You might want to return a dictionary mapping each action in
        # `self.action_set` to the probability of picking that action
        raise NotImplementedError

    def observe_utility(self, utility):
        assert isinstance(utility, dict) and utility.keys() == self.action_set

        # FINISH
        raise NotImplementedError

"""

class Cfr(object):
    def __init__(self, tfsdp, rm_class=RegretMatching):

        __slots__ = ('tfsdp', 'J', 'local_regret_minimizers', 'strategy_sum')
        
        self.tfsdp = tfsdp
        self.J = [node for node in tfsdp if node["type"] == "decision"]
        
        self.local_regret_minimizers: Map[RM_ID, rm_class] = {j["id"]: rm_class(j["actions"]) for j in self.J}
        self.strategy_sum: Map[Sequence, cumulate] = {(j["id"], action): 0 for j in self.J for action in j["actions"]}

    def next_strategy(self) -> Map[Sequence, prob]:
        
        local_strats = {rm_id: rm.next_strategy() for rm_id, rm in self.local_regret_minimizers.items()}

        strategy: Map[Sequence, prob] = {}
        for j in self.J:
            match j["parent_sequence"]:
                case None: reach_prob = 1.0
                case _:    reach_prob = strategy[j["parent_sequence"]]
            for a in j["actions"]:
                strategy[(j["id"], a)] = reach_prob * local_strats[j["id"]][a]
        return strategy

    def observe_utility(self, utility: Utility):
        p = lambda v, a: succ(self.tfsdp, v, a)
        local_strats = {rm_id: rm.next_strategy() for rm_id, rm in self.local_regret_minimizers.items()}
        
        EV: Map[Node, val] = {None: 0}
        for j in reversed(self.tfsdp):
            EV[j["id"]] = 0
            match j["type"]:
                case "decision": 
                    local_strat = local_strats[j["id"]]
                    for action in j["actions"]:
                        EV[j["id"]] += local_strat[action] * (utility[(j["id"], action)] + EV[p(j["id"], action)])
                case "observation":
                    for signal in j["signals"]:
                        EV[j["id"]] += EV[p(j["id"], signal)]
        # we have computed the average of the counterfactual values

        for j in self.J:
                local_utility: Map[Action, val] = {action: (utility[(j["id"], action)] + EV[p(j["id"], action)]) 
                    for action in j["actions"]}
                self.local_regret_minimizers[j["id"]].observe_utility(local_utility)


def solve_problem_3_1(game):
    tfsdp1, tfsdp2 = game["decision_problem_pl1"], game["decision_problem_pl2"]
    T = 10_000
    
    p1_cfr = Cfr(tfsdp=tfsdp1)

    y = uniform_sf_strategy(tfsdp2)
    utility = compute_utility_vector_pl1(game, y)

    strategy_sum = {sequence: 0 for sequence in get_sequence_set(tfsdp1)}
    for _ in range(T):
        x = p1_cfr.next_strategy()
        p1_cfr.observe_utility(utility)

        for sequence in get_sequence_set(tfsdp1):
            strategy_sum[sequence] += x[sequence]
    average_strategy = {sequence: strategy/T for sequence, strategy in strategy_sum.items()}
    print(f"ev x_starAy={expected_utility_pl1(game, average_strategy, y)}, br value={best_response_value(tfsdp1, utility)}")


def solve_problem_3_2(game):
    # FINISH
    raise NotImplementedError


def solve_problem_3_3(game):
    # FINISH
    raise NotImplementedError


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