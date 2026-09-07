"""
Model-free / online-learning baselines for RMAB experiments.

These policies do not use the true transition matrix P or reward matrix R when
making decisions or updating their internal tables. The simulator still uses the
true bandit as the hidden environment, exactly as a physical system would.

Known-model oracle policies are included only as reference curves.
"""

import numpy as np

import bandit_lp
import strategies


class UnknownModelPolicy:
    """Interface for online policies that act on explicit arm states."""

    def __init__(self, S, A, N, alpha):
        if A != 2:
            raise ValueError("The current benchmark assumes two actions.")
        self.S = int(S)
        self.A = int(A)
        self.N = int(N)
        self.alpha = float(alpha)
        self.budget = int(self.alpha * self.N)
        self.rng = np.random.default_rng(0)
        self.t = 0

    def reset(self, seed):
        self.rng = np.random.default_rng(seed)
        self.t = 0

    def select_actions(self, states):
        raise NotImplementedError

    def update(self, states, actions, rewards, next_states):
        del states, actions, rewards, next_states
        self.t += 1

    def _random_actions(self):
        actions = np.zeros(self.N, dtype=int)
        if self.budget > 0:
            selected = self.rng.choice(
                self.N,
                size=self.budget,
                replace=False,
            )
            actions[selected] = 1
        return actions

    def _activate_largest_scores(self, scores):
        actions = np.zeros(self.N, dtype=int)
        if self.budget == 0:
            return actions

        scores = np.asarray(scores, dtype=float)
        tie_breaker = self.rng.uniform(0.0, 1e-12, size=self.N)
        selected = np.argpartition(
            scores + tie_breaker,
            -self.budget,
        )[-self.budget :]
        actions[selected] = 1
        return actions


class OnlineRandomPolicy(UnknownModelPolicy):
    """Unknown-model baseline: uniformly activate budget arms each period."""

    def select_actions(self, states):
        del states
        return self._random_actions()


class OnlineRewardGreedyPolicy(UnknownModelPolicy):
    """
    Learn immediate rewards only, then greedily rank states by R_hat(s,1)-R_hat(s,0).

    This baseline ignores transition effects. It is useful because it separates
    "learning rewards" from "learning long-term dynamic value".
    """

    def __init__(
        self,
        S,
        A,
        N,
        alpha,
        epsilon_start=0.30,
        epsilon_min=0.02,
        optimistic_value=1.0,
    ):
        super().__init__(S, A, N, alpha)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_min = float(epsilon_min)
        self.optimistic_value = float(optimistic_value)
        self.reward_sum = None
        self.reward_count = None

    def reset(self, seed):
        super().reset(seed)
        self.reward_sum = np.zeros((self.S, self.A))
        self.reward_count = np.zeros((self.S, self.A))

    def _epsilon(self):
        return max(self.epsilon_min, self.epsilon_start / np.sqrt(1 + self.t / 50))

    def _mean_reward(self, state, action):
        count = self.reward_count[state, action]
        if count <= 0:
            return self.optimistic_value
        return self.reward_sum[state, action] / count

    def select_actions(self, states):
        if self.rng.random() < self._epsilon():
            return self._random_actions()

        scores = np.array(
            [
                self._mean_reward(s, 1) - self._mean_reward(s, 0)
                for s in states
            ]
        )
        return self._activate_largest_scores(scores)

    def update(self, states, actions, rewards, next_states):
        del next_states
        for state, action, reward in zip(states, actions, rewards):
            self.reward_sum[state, action] += reward
            self.reward_count[state, action] += 1
        self.t += 1


class OnlineUCBRewardPolicy(OnlineRewardGreedyPolicy):
    """
    Reward-learning baseline with an upper-confidence bonus.

    The bonus is attached mainly to the active action so that states with little
    active-sampling history are explored.
    """

    def __init__(self, S, A, N, alpha, ucb_scale=1.0):
        super().__init__(S, A, N, alpha, epsilon_start=0.0, epsilon_min=0.0)
        self.ucb_scale = float(ucb_scale)

    def _ucb_mean(self, state, action):
        count = self.reward_count[state, action]
        if count <= 0:
            return self.optimistic_value + self.ucb_scale

        total_count = max(1.0, float(np.sum(self.reward_count)))
        mean = self.reward_sum[state, action] / count
        bonus = self.ucb_scale * np.sqrt(np.log(total_count + 1.0) / count)
        return mean + bonus

    def select_actions(self, states):
        scores = np.array(
            [
                self._ucb_mean(s, 1) - self._mean_reward(s, 0)
                for s in states
            ]
        )
        return self._activate_largest_scores(scores)


class OnlineQLearningIndexPolicy(UnknownModelPolicy):
    """
    Model-free Q-index policy.

    It learns one shared tabular Q table from all arm observations:

        Q(s,a) <- Q(s,a) + eta [r + gamma max_a' Q(s',a') - Q(s,a)].

    The online priority score is Q(s,1)-Q(s,0). No P or R matrix is used by the
    policy; it only sees sampled states, actions, rewards, and next states.
    """

    def __init__(
        self,
        S,
        A,
        N,
        alpha,
        gamma=0.95,
        epsilon_start=0.30,
        epsilon_min=0.02,
        active_penalty=0.0,
    ):
        super().__init__(S, A, N, alpha)
        self.gamma = float(gamma)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_min = float(epsilon_min)
        self.active_penalty = float(active_penalty)
        self.Q = None
        self.visit_count = None

    def reset(self, seed):
        super().reset(seed)
        self.Q = np.zeros((self.S, self.A))
        self.visit_count = np.zeros((self.S, self.A))

    def _epsilon(self):
        return max(self.epsilon_min, self.epsilon_start / np.sqrt(1 + self.t / 50))

    def select_actions(self, states):
        if self.rng.random() < self._epsilon():
            return self._random_actions()

        scores = self.Q[states, 1] - self.Q[states, 0]
        return self._activate_largest_scores(scores)

    def update(self, states, actions, rewards, next_states):
        for state, action, reward, next_state in zip(
            states,
            actions,
            rewards,
            next_states,
        ):
            self.visit_count[state, action] += 1
            learning_rate = min(
                0.5,
                1.0 / (self.visit_count[state, action] ** 0.6),
            )
            adjusted_reward = reward - self.active_penalty * int(action == 1)
            target = adjusted_reward + self.gamma * np.max(self.Q[next_state])
            self.Q[state, action] += learning_rate * (
                target - self.Q[state, action]
            )

        self.t += 1


class OnlinePlugInWhittlePolicy(UnknownModelPolicy):
    """
    Learn an empirical model P_hat,R_hat, then plug it into Whittle.

    This is the RMAB-specific unknown-model baseline:

    1. collect samples (s, a, r, s_next);
    2. estimate transition probabilities P_hat and rewards R_hat;
    3. compute Whittle indices for the estimated single-arm model;
    4. activate the arms with the largest estimated indices.

    The policy never reads the true P or R. It only uses the learned empirical
    model as if it were the true model.
    """

    def __init__(
        self,
        S,
        A,
        N,
        alpha,
        recompute_interval=50,
        epsilon_start=0.30,
        epsilon_min=0.02,
        transition_prior=1.0,
        reward_prior=0.0,
        reward_prior_count=1.0,
    ):
        super().__init__(S, A, N, alpha)
        self.recompute_interval = int(recompute_interval)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_min = float(epsilon_min)
        self.transition_prior = float(transition_prior)
        self.reward_prior = float(reward_prior)
        self.reward_prior_count = float(reward_prior_count)
        self.transition_count = None
        self.reward_sum = None
        self.reward_count = None
        self.indices = None
        self.num_index_recomputations = 0
        self.last_recompute_status = "not_started"

    def reset(self, seed):
        super().reset(seed)
        self.transition_count = np.ones(
            (self.S, self.A, self.S),
            dtype=float,
        ) * self.transition_prior
        self.reward_count = np.ones((self.S, self.A), dtype=float) * (
            self.reward_prior_count
        )
        self.reward_sum = self.reward_count * self.reward_prior
        self.indices = np.zeros(self.S)
        self.num_index_recomputations = 0
        self.last_recompute_status = "initialized"
        self._recompute_indices()

    def _epsilon(self):
        return max(self.epsilon_min, self.epsilon_start / np.sqrt(1 + self.t / 50))

    def _estimate_model(self):
        P_hat = self.transition_count / self.transition_count.sum(
            axis=2,
            keepdims=True,
        )
        R_hat = np.divide(
            self.reward_sum,
            self.reward_count,
            out=np.zeros_like(self.reward_sum),
            where=self.reward_count > 0,
        )
        return P_hat, R_hat

    def _recompute_indices(self):
        P_hat, R_hat = self._estimate_model()
        estimated_bandit = bandit_lp.BanditInstance(P_hat, R_hat)

        try:
            teacher_policy = strategies.WhittleIndexStrategy(
                estimated_bandit,
                self.alpha,
            )
            indices = np.asarray(teacher_policy.whittle_indices, dtype=float)
            if not np.all(np.isfinite(indices)):
                raise ValueError("Estimated Whittle indices contain NaN/inf.")
            self.indices = indices
            self.last_recompute_status = "whittle"
        except Exception:
            # Some early empirical models may be numerically awkward. Falling
            # back to immediate reward advantage keeps the online experiment
            # running while more data are collected.
            self.indices = R_hat[:, 1] - R_hat[:, 0]
            self.last_recompute_status = "reward_advantage_fallback"

        self.num_index_recomputations += 1

    def select_actions(self, states):
        if self.rng.random() < self._epsilon():
            return self._random_actions()
        return self._activate_largest_scores(self.indices[states])

    def update(self, states, actions, rewards, next_states):
        for state, action, reward, next_state in zip(
            states,
            actions,
            rewards,
            next_states,
        ):
            self.transition_count[state, action, next_state] += 1.0
            self.reward_sum[state, action] += reward
            self.reward_count[state, action] += 1.0

        self.t += 1
        if self.t % self.recompute_interval == 0:
            self._recompute_indices()


class KnownWhittleOraclePolicy(UnknownModelPolicy):
    """
    Known-model Whittle policy used only as a reference curve.

    This policy does use P,R, so it should not be classified as unknown-model.
    """

    def __init__(self, bandit, N, alpha):
        super().__init__(bandit.S, bandit.A, N, alpha)
        teacher_policy = strategies.WhittleIndexStrategy(bandit, alpha)
        self.indices = np.asarray(teacher_policy.whittle_indices, dtype=float)

    def select_actions(self, states):
        return self._activate_largest_scores(self.indices[states])


class KnownLPPriorityOraclePolicy(UnknownModelPolicy):
    """Known-model LP-Priority reference curve."""

    def __init__(self, bandit, N, alpha):
        super().__init__(bandit.S, bandit.A, N, alpha)
        teacher_policy = strategies.LPPriorityStrategy(bandit, alpha)
        self.indices = np.asarray(teacher_policy.lp_index, dtype=float)

    def select_actions(self, states):
        return self._activate_largest_scores(self.indices[states])


UNKNOWN_POLICY_NAMES = [
    "OnlinePlugInWhittle",
    "OnlineQLearningIndex",
    "OnlineUCBReward",
    "OnlineRewardGreedy",
    "OnlineRandom",
]


ORACLE_POLICY_NAMES = [
    "KnownWhittleOracle",
    "KnownLPPriorityOracle",
]


def make_unknown_model_policy(policy_name, bandit, N, alpha):
    """Create either an unknown-model online policy or a known-model oracle."""
    if policy_name == "OnlinePlugInWhittle":
        return OnlinePlugInWhittlePolicy(bandit.S, bandit.A, N, alpha)
    if policy_name == "OnlineQLearningIndex":
        return OnlineQLearningIndexPolicy(bandit.S, bandit.A, N, alpha)
    if policy_name == "OnlineUCBReward":
        return OnlineUCBRewardPolicy(bandit.S, bandit.A, N, alpha)
    if policy_name == "OnlineRewardGreedy":
        return OnlineRewardGreedyPolicy(bandit.S, bandit.A, N, alpha)
    if policy_name == "OnlineRandom":
        return OnlineRandomPolicy(bandit.S, bandit.A, N, alpha)
    if policy_name == "KnownWhittleOracle":
        return KnownWhittleOraclePolicy(bandit, N, alpha)
    if policy_name == "KnownLPPriorityOracle":
        return KnownLPPriorityOraclePolicy(bandit, N, alpha)

    raise ValueError(f"Unknown online policy: {policy_name}")


def simulate_unknown_model(
    bandit,
    policy,
    initial_state,
    N,
    horizon,
    seed,
    tail_fraction=0.25,
):
    """
    Explicit-arm simulator for unknown-model policies.

    The policy receives only samples. The hidden environment uses bandit.P and
    bandit.R to generate rewards and next states.
    """
    rng = np.random.default_rng(seed)
    policy.reset(seed + 1)

    states = rng.choice(bandit.S, size=N, p=initial_state)
    reward_history = np.zeros(horizon)

    for t in range(horizon):
        actions = np.asarray(policy.select_actions(states), dtype=int)
        if actions.shape != (N,):
            raise ValueError("Policy returned an action vector with wrong shape.")
        if np.sum(actions) != policy.budget:
            raise ValueError("Policy did not satisfy the exact activation budget.")

        rewards = np.empty(N)
        next_states = np.empty(N, dtype=int)

        for i in range(N):
            state = states[i]
            action = actions[i]
            rewards[i] = bandit.R[state, action]
            next_states[i] = rng.choice(
                bandit.S,
                p=bandit.P[state, action, :],
            )

        policy.update(states, actions, rewards, next_states)
        reward_history[t] = np.mean(rewards)
        states = next_states

    tail_start = int((1.0 - tail_fraction) * horizon)
    return {
        "mean_reward": float(np.mean(reward_history)),
        "tail_mean_reward": float(np.mean(reward_history[tail_start:])),
        "reward_history": reward_history,
        "final_states": states,
    }
