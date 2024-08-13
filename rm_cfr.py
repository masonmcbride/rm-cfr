# ##############################################################################
# I'm gonna give credit of most of this code to Gabriele Farina.
# Source is CS 15-888 at CMU. Taught by Tumoas Sandholm and GSI Gabriele Farina (https://www.cs.cmu.edu/~sandholm/cs15-888F21/)
# This implements CFR using Local Regret Minimzers for each decision node of the tree form sequential decision problem

import os

History = str
Action = str
Sequence = tuple[History, Action]
Prob = float
Node = str

class RegretMatching(object):
    def __init__(self, action_set):
        self.action_set = set(action_set)
        self.regret_sum: dict[Action, float] = {action: 0 for action in self.action_set}

    def next_strategy(self) -> dict[str,float]:
        r_plus: dict[str,float] = {action: max(regret, 0) for action, regret in self.regret_sum.items()}
        if not any(r_plus.values()):
            return {action: 1./len(self.action_set) for action in self.action_set}
        else:
            normalizer = sum(r_plus.values())
            return {action: regret/normalizer for action, regret in r_plus.items()}

    def observe_utility(self, utility: dict[str,float]):
        assert isinstance(utility, dict) and utility.keys() == self.action_set

        x: dict[str,float] = self.next_strategy()
        ground_truth: float = sum(x[action] * utility[action] for action in self.action_set)

        self.regret_sum = {action: current_sum + utility[action] - ground_truth for action,current_sum in self.regret_sum.items()}
class RegretMatchingPlus(object):
    def __init__(self, action_set):
        self.action_set = set(action_set)
        self.regret_sum = {action: 0 for action in self.action_set}

    def next_strategy(self):
        r_plus = {action: max(regret, 0) for action,regret in self.regret_sum.items()}
        if not any(r_plus.values()):
            return {action: 1./len(self.action_set) for action in self.action_set}
        else:
            normalizer = sum(r_plus.values())
            return {action: regret/normalizer for action,regret in r_plus.items()}

    def observe_utility(self, utility):
        assert isinstance(utility, dict) and utility.keys() == self.action_set

        x = self.next_strategy()
        ground_truth: float = sum(x[action] * utility[action] for action in self.action_set)
        
        self.regret_sum = {action: max(current_sum + utility[action] - ground_truth,0) for action,current_sum in self.regret_sum.items()}

class Cfr(object):
    def __init__(self, tfsdp, rm_class=RegretMatching):

        __slots__ = ('tfsdp', 'J', 'local_regret_minimizers')
        
        # tree form sequential decision problem 
        self.tfsdp = tfsdp

        # decision nodes 
        self.J = [node for node in tfsdp if node["type"] == "decision"]
        
        # regeret minimizers assigned to each decision node
        self.local_regret_minimizers: dict[str, RegretMatching] = {j["id"]: rm_class(j["actions"]) for j in self.J}

    def next_strategy(self) -> dict[Sequence, Prob]:
        """Based on the current regret minmizers, return the strategy for each (history,action) pair """
        
        # get the local strategy at each regret minimizer 
        local_strats = {rm_id: rm.next_strategy() for rm_id, rm in self.local_regret_minimizers.items()}

        # go through each decision node top-down and mulitply by the reach_prob
        strategy = {}
        for j in self.J:
            match j["parent_sequence"]:
                case None: reach_prob = 1.
                case parent_sequence: reach_prob = strategy[parent_sequence]
            for a in j["actions"]:
                strategy[(j["id"], a)] = reach_prob * local_strats[j["id"]][a]
        return strategy

    def observe_utility(self, utility):
        s = lambda v, a: succ(self.tfsdp, v, a)
        local_strats = {rm_id: rm.next_strategy() for rm_id, rm in self.local_regret_minimizers.items()}
        
        # go through each node bottom-up and build up expected value for each node 
        EV: dict[Node, float] = {None: 0}
        for j in reversed(self.tfsdp):
            EV[j["id"]] = 0
            match j["type"]:
                case "decision": 
                    local_strat = local_strats[j["id"]]
                    for action in j["actions"]:
                        EV[j["id"]] += local_strat[action] * (utility[(j["id"], action)] + EV[s(j["id"], action)])
                case "observation":
                    for signal in j["signals"]:
                        EV[j["id"]] += EV[s(j["id"], signal)]
        # we have computed the average of the counterfactual values

        for j in self.J:
                local_utility: dict[Action, float] = {action: utility[(j["id"], action)] + EV[s(j["id"], action)]
                    for action in j["actions"]}
                self.local_regret_minimizers[j["id"]].observe_utility(local_utility)

def Cfr_plus(tfsdp) -> Cfr:
    return Cfr(tfsdp=tfsdp,rm_class=RegretMatchingPlus)

def get_sequence_set(tfsdp: dict) -> set[Sequence]:
    """Returns a set of all sequences in the given tree-form sequential decision
    process (TFSDP)"""

    sequences = {(node['id'],action) for node in tfsdp if node['type'] == 'decision' for action in node['actions']}
    return sequences

def is_valid_RSigma_vector(tfsdp: dict, obj):
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
    """Returns Ay, where A is the payoff matrix of the game and y is
    the given strategy for Player 2"""

    assert_is_valid_sf_strategy(game['decision_problem_pl2'], sf_strategy_pl2)

    sequence_set = get_sequence_set(game['decision_problem_pl1'])
    utility = {sequence: 0 for sequence in sequence_set}
    for entry in game["utility_pl1"]:
        utility[entry["sequence_pl1"]] += entry["value"] * sf_strategy_pl2[entry["sequence_pl2"]]

    assert is_valid_RSigma_vector(game["decision_problem_pl1"], utility)
    return utility

def compute_utility_vector_pl2(game, sf_strategy_pl1):
    """Returns -A^Tx, where A is the payoff matrix of the
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

def succ(tfsdp, history: str, action: str):
    """Return node id that is the successor of the (history,action) tuple"""
    for node in tfsdp:
        if node["parent_edge"] == (history, action):
            return node["id"]
    return None

def expected_utility_pl1(game, sf_strategy_pl1: dict[Sequence, Prob], sf_strategy_pl2: dict[Sequence, Prob]):
    """Returns the expected utility for Player 1 in the game, when the two
    players play according to the given strategies"""

    assert_is_valid_sf_strategy(game["decision_problem_pl1"], sf_strategy_pl1)
    assert_is_valid_sf_strategy(game["decision_problem_pl2"], sf_strategy_pl2)

    Ay: dict[Sequence, float] = compute_utility_vector_pl1(game, sf_strategy_pl2)
    xAy = sum([sf_strategy_pl1[sequence]*Ay[sequence] for sequence in sf_strategy_pl1])
    return xAy

def uniform_sf_strategy(tfsdp) -> dict[tuple[str,str], float]:
    """Returns the uniform sequence-form strategy for the given tree-form
    sequential decision process"""

    uniform_strategy: dict[tuple[str,str], float] = {}
    J = [node for node in tfsdp if node["type"] == "decision"]
    for j in J:
        match j["parent_sequence"]:
            case None: parent_reach = 1
            case parent_sequence: parent_reach = uniform_strategy[parent_sequence]
        for action in j["actions"]:
            uniform_strategy[(j["id"], action)] = parent_reach/len(j["actions"])

    assert_is_valid_sf_strategy(tfsdp, uniform_strategy)
    return uniform_strategy