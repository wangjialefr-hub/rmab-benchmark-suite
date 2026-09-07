"""
Policy factory for RMAB benchmark experiments.

This file is your own layer. It does not modify the teacher implementations in
bandit_lp.py or strategies.py. It only creates policy objects in a consistent
way, including a few simple baseline policies.
"""

import hashlib

import numpy as np


POLICY_NAMES = [
    "WhittleIndexStrategy",
    "LPPriorityStrategy",
    "FTVA_Strategy",
    "LPupdateStrategy",
    "Myopic",
    "RandomPriority",
    "RoundRobin",
    "LPRandomized",
    "QWhittleKnownModel",
]


POLICY_ALIASES = {
    "Whittle": "WhittleIndexStrategy",
    "WhittleIndex": "WhittleIndexStrategy",
    "LP-Priority": "LPPriorityStrategy",
    "LPPriority": "LPPriorityStrategy",
    "FTVA": "FTVA_Strategy",
    "LP-Update": "LPupdateStrategy",
    "LPUpdate": "LPupdateStrategy",
    "Random-Priority": "RandomPriority",
    "Round-Robin": "RoundRobin",
    "RoundRobinStrategy": "RoundRobin",
    "LP-Randomized": "LPRandomized",
    "QWhittle": "QWhittleKnownModel",
    "RLWhittle": "QWhittleKnownModel",
    "Q-Learning-Whittle": "QWhittleKnownModel",
}


QWHITTLE_GAMMA = 0.95
QWHITTLE_NUM_PENALTIES = 41
QWHITTLE_STEPS_PER_PENALTY = 50000
QWHITTLE_TRAINING_SEED = 2026

_QWHITTLE_PRIORITY_CACHE = {}


def canonical_policy_name(policy_name):
    """Convert short aliases to the internal policy name."""
    return POLICY_ALIASES.get(policy_name, policy_name)


def _import_strategies():
    """
    Import the teacher strategies module.

    In the notebook, make sure CODE_DIR has already been inserted into sys.path:

        sys.path.insert(0, str(CODE_DIR))
    """
    try:
        import strategies
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import strategies.py. Add the teacher code folder to "
            "sys.path before calling make_policy()."
        ) from exc
    return strategies


class LPRandomizedStrategy:
    """
    Simple baseline based on the relaxed LP solution.

    The relaxed LP gives y_star. We convert it into a per-state activation
    probability:

        pi_star[s] = y_star[s, 1] / (y_star[s, 0] + y_star[s, 1])

    For the current empirical state distribution x, we first set:

        active[s] = x[s] * pi_star[s]

    Then we deterministically repair the active mass so that the total active
    fraction equals alpha.
    """

    def __init__(self, bandit, alpha):
        self.bandit = bandit
        self.alpha = alpha
        self.X = None
        self.reward = None

        _, y_star, _, _ = bandit.relaxed_lp_average_reward(alpha)
        x_star = np.sum(y_star, axis=1)
        self.pi_star = np.divide(
            y_star[:, 1],
            x_star,
            out=np.zeros_like(x_star),
            where=x_star > 1e-12,
        )

    def hashname(self):
        h = hashlib.new("sha256")
        h.update(b"lp-randomized")
        h.update(self.bandit.hashname().encode())
        h.update(str(self.alpha).encode())
        return h.hexdigest()[0:10]

    def next_y(self, state_x, N=None):
        state_x = np.asarray(state_x, dtype=float)
        active = state_x * self.pi_star
        active = self._repair_budget(active, state_x)

        y = np.zeros((self.bandit.S, self.bandit.A))
        y[:, 1] = active
        y[:, 0] = state_x - active
        return y

    def _repair_budget(self, active, state_x):
        active = np.clip(active, 0.0, state_x)
        target = min(self.alpha, np.sum(state_x))
        diff = target - np.sum(active)

        if abs(diff) <= 1e-12:
            return active

        if diff > 0:
            # Add activation to high-pi states first.
            for s in np.flip(np.argsort(self.pi_star)):
                add = min(diff, state_x[s] - active[s])
                active[s] += add
                diff -= add
                if diff <= 1e-12:
                    break
        else:
            # Remove activation from low-pi states first.
            diff = -diff
            for s in np.argsort(self.pi_star):
                remove = min(diff, active[s])
                active[s] -= remove
                diff -= remove
                if diff <= 1e-12:
                    break

        return active


class RoundRobinStrategy:
    """
    Round-robin baseline over arm identities.

    This policy ignores P, R, and the current state quality. It activates a fixed
    budget of arms by cycling through arm indices:

        0, 1, ..., budget-1
        budget, ..., 2*budget-1
        ...

    Because strategies.simulate() only passes the empirical state distribution
    state_x, this class internally maintains the individual states of N arms and
    simulates their transitions after each action, like FTVA_Strategy does.
    """

    def __init__(self, bandit, alpha):
        self.bandit = bandit
        self.alpha = alpha
        self.X = None
        self.reward = None
        self.arm_states = None
        self.pointer = 0

    def hashname(self):
        h = hashlib.new("sha256")
        h.update(b"round-robin")
        h.update(self.bandit.hashname().encode())
        h.update(str(self.alpha).encode())
        return h.hexdigest()[0:10]

    def next_y(self, state_x, N=None):
        assert isinstance(N, int), "RoundRobin is only defined for finite N."

        if self.arm_states is None or len(self.arm_states) != N:
            self.arm_states = self._initialize_arm_states(state_x, N)
            self.pointer = 0

        budget = int(self.alpha * N)
        budget = min(max(budget, 0), N)

        actions = np.zeros(N, dtype=int)
        if budget > 0:
            active_indices = (self.pointer + np.arange(budget)) % N
            actions[active_indices] = 1
            self.pointer = (self.pointer + budget) % N

        y = self._y_from_states_actions(self.arm_states, actions)
        self.reward = np.tensordot(y, self.bandit.R)

        for i in range(N):
            s = self.arm_states[i]
            a = actions[i]
            self.arm_states[i] = np.random.choice(self.bandit.S, p=self.bandit.P[s, a, :])

        self.X = self._x_from_states(self.arm_states)
        return y

    def _initialize_arm_states(self, state_x, N):
        state_x = np.asarray(state_x, dtype=float)
        states = np.zeros(N, dtype=int)

        n = 0
        for s in range(self.bandit.S):
            count = int(N * state_x[s])
            states[n:n + count] = s
            n += count

        if n < N:
            residual = np.maximum(state_x, 0.0)
            residual = residual / residual.sum()
            states[n:] = np.random.choice(self.bandit.S, size=N - n, p=residual)

        return states

    def _x_from_states(self, states):
        x = np.zeros(self.bandit.S)
        for s in states:
            x[s] += 1
        return x / len(states)

    def _y_from_states_actions(self, states, actions):
        y = np.zeros((self.bandit.S, self.bandit.A))
        for s, a in zip(states, actions):
            y[s, a] += 1
        return y / len(states)


def _qwhittle_cache_key(
    bandit,
    *,
    gamma=QWHITTLE_GAMMA,
    num_penalties=QWHITTLE_NUM_PENALTIES,
    steps_per_penalty=QWHITTLE_STEPS_PER_PENALTY,
    training_seed=QWHITTLE_TRAINING_SEED,
):
    return (
        bandit.hashname(),
        float(gamma),
        int(num_penalties),
        int(steps_per_penalty),
        int(training_seed),
    )


def _train_single_arm_q_values(
    bandit,
    active_penalty,
    rng,
    *,
    gamma=QWHITTLE_GAMMA,
    steps=QWHITTLE_STEPS_PER_PENALTY,
):
    """
    Tabular Q-learning on the single-arm MDP with an active-action penalty.

    The known P,R model is used only as a simulator: transitions are sampled
    from P[s,a,:], and rewards are R[s,a] - active_penalty * 1{a=1}.
    """
    Q = np.zeros((bandit.S, bandit.A))
    visits = np.zeros((bandit.S, bandit.A))
    state = int(rng.integers(bandit.S))

    for t in range(steps):
        epsilon = max(0.02, 0.30 * (1.0 - t / steps))

        if rng.random() < 0.05:
            state = int(rng.integers(bandit.S))

        if rng.random() < epsilon:
            action = int(rng.integers(bandit.A))
        else:
            best_actions = np.flatnonzero(Q[state] == np.max(Q[state]))
            action = int(rng.choice(best_actions))

        next_state = int(rng.choice(bandit.S, p=bandit.P[state, action, :]))
        reward = bandit.R[state, action] - active_penalty * int(action == 1)

        visits[state, action] += 1
        learning_rate = min(0.5, 1.0 / (visits[state, action] ** 0.6))
        target = reward + gamma * np.max(Q[next_state])
        Q[state, action] += learning_rate * (target - Q[state, action])

        state = next_state

    return Q


def _learn_qwhittle_indices(
    bandit,
    *,
    gamma=QWHITTLE_GAMMA,
    num_penalties=QWHITTLE_NUM_PENALTIES,
    steps_per_penalty=QWHITTLE_STEPS_PER_PENALTY,
    training_seed=QWHITTLE_TRAINING_SEED,
):
    """
    Learn approximate Whittle-style indices by Q-learning.

    For each active-action penalty lambda, Q-learning estimates the single-arm
    Q-values. The learned index of state s is the lambda where:

        Q_lambda(s, active) - Q_lambda(s, passive) ~= 0.

    This is a model-based RL baseline because the known P,R model is used as a
    simulator to generate training transitions.
    """
    cache_key = _qwhittle_cache_key(
        bandit,
        gamma=gamma,
        num_penalties=num_penalties,
        steps_per_penalty=steps_per_penalty,
        training_seed=training_seed,
    )
    if cache_key in _QWHITTLE_PRIORITY_CACHE:
        return _QWHITTLE_PRIORITY_CACHE[cache_key]

    reward_scale = max(1.0, float(np.max(np.abs(bandit.R))))
    penalty_scale = 5.0 * reward_scale
    penalties = np.linspace(-penalty_scale, penalty_scale, num_penalties)

    rng = np.random.default_rng(training_seed)
    advantages = np.zeros((num_penalties, bandit.S))

    for i, penalty in enumerate(penalties):
        Q = _train_single_arm_q_values(
            bandit,
            active_penalty=penalty,
            rng=rng,
            gamma=gamma,
            steps=steps_per_penalty,
        )
        advantages[i] = Q[:, 1] - Q[:, 0]

    indices = np.zeros(bandit.S)
    for s in range(bandit.S):
        adv = advantages[:, s]
        crossing = np.where((adv[:-1] >= 0.0) & (adv[1:] <= 0.0))[0]

        if len(crossing) > 0:
            k = int(crossing[0])
            x0, x1 = penalties[k], penalties[k + 1]
            y0, y1 = adv[k], adv[k + 1]
            if abs(y1 - y0) <= 1e-12:
                indices[s] = 0.5 * (x0 + x1)
            else:
                indices[s] = x0 - y0 * (x1 - x0) / (y1 - y0)
        elif np.all(adv > 0.0):
            indices[s] = penalties[-1]
        elif np.all(adv < 0.0):
            indices[s] = penalties[0]
        else:
            indices[s] = penalties[int(np.argmin(np.abs(adv)))]

    priority = np.flip(np.argsort(indices))
    result = {
        "indices": indices,
        "priority": priority,
        "penalties": penalties,
        "advantages": advantages,
    }
    _QWHITTLE_PRIORITY_CACHE[cache_key] = result
    return result


def make_policy(policy_name, bandit, alpha, lp_update_horizon=20, random_seed=0):
    """
    Create one policy object.

    Parameters
    ----------
    policy_name:
        Name of the policy. Both long names and aliases are supported.
    bandit:
        A BanditInstance from bandit_lp.py.
    alpha:
        Activation budget fraction.
    lp_update_horizon:
        Rolling planning horizon used only by LPupdateStrategy.
    random_seed:
        Seed used only by RandomPriority.
    """
    strategies = _import_strategies()
    policy_name = canonical_policy_name(policy_name)

    if policy_name == "WhittleIndexStrategy":
        return strategies.WhittleIndexStrategy(bandit, alpha)

    if policy_name == "LPPriorityStrategy":
        return strategies.LPPriorityStrategy(bandit, alpha)

    if policy_name == "FTVA_Strategy":
        return strategies.FTVA_Strategy(bandit, alpha)

    if policy_name == "LPupdateStrategy":
        return strategies.LPupdateStrategy(
            bandit,
            alpha,
            time=lp_update_horizon,
        )

    if policy_name == "Myopic":
        # Prioritize states by immediate reward gain from action 1 over action 0.
        priority = np.flip(np.argsort(bandit.R[:, 1] - bandit.R[:, 0]))
        return strategies.PriorityStrategy(priority, alpha)

    if policy_name == "RandomPriority":
        # Fixed random priority order. Useful as a weak baseline.
        rng = np.random.default_rng(random_seed)
        priority = rng.permutation(bandit.S)
        return strategies.PriorityStrategy(priority, alpha)

    if policy_name == "RoundRobin":
        return RoundRobinStrategy(bandit, alpha)

    if policy_name == "LPRandomized":
        return LPRandomizedStrategy(bandit, alpha)

    if policy_name == "QWhittleKnownModel":
        learned = _learn_qwhittle_indices(bandit)
        return strategies.PriorityStrategy(learned["priority"], alpha)

    raise ValueError(f"Unknown policy: {policy_name}")


def can_reuse_policy(policy_name):
    """
    Whether it is safe to reuse a policy object across simulations.

    FTVA and RoundRobin store internal simulated states, so rebuild them for
    each Monte Carlo replication. The others are deterministic mappings,
    possibly with harmless memoization.
    """
    return canonical_policy_name(policy_name) not in {"FTVA_Strategy", "RoundRobin"}
