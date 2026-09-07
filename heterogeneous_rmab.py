"""
Core tools for known-model heterogeneous RMAB experiments.

Unlike the teacher's homogeneous simulator, this module keeps every arm
explicitly:

    arm i has its own P_i, R_i, and current state states[i].

The existing teacher files are reused for BanditInstance and Whittle-index
computation, but they are not modified.
"""

from dataclasses import dataclass

import numpy as np
import pulp

import strategies
from known_model_extra_instances import (
    maintenance_degradation_instance,
    wireless_channel_instance,
)


@dataclass
class HeterogeneousRMAB:
    """A collection of arms with possibly different transition/reward models."""

    name: str
    arms: list
    alpha: float
    initial_distributions: list
    type_ids: np.ndarray
    arm_parameters: list
    source: str

    def __post_init__(self):
        if not self.arms:
            raise ValueError("A heterogeneous RMAB must contain at least one arm.")
        if len(self.initial_distributions) != len(self.arms):
            raise ValueError("One initial distribution is required for each arm.")
        if len(self.type_ids) != len(self.arms):
            raise ValueError("type_ids must have one entry per arm.")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must lie strictly between 0 and 1.")

        for i, (arm, initial) in enumerate(
            zip(self.arms, self.initial_distributions)
        ):
            if arm.A != 2:
                raise ValueError(f"Arm {i} does not have two actions.")
            initial = np.asarray(initial, dtype=float)
            if initial.shape != (arm.S,):
                raise ValueError(f"Initial distribution has wrong shape for arm {i}.")
            if np.any(initial < 0) or not np.isclose(initial.sum(), 1.0):
                raise ValueError(f"Invalid initial distribution for arm {i}.")

    @property
    def N(self):
        return len(self.arms)

    @property
    def budget(self):
        return int(self.alpha * self.N)


class HeterogeneousPolicy:
    """Small policy interface used by the explicit-arm simulator."""

    def __init__(self, environment):
        self.environment = environment
        self.N = environment.N
        self.budget = environment.budget
        self.rng = np.random.default_rng(0)

    def reset(self, seed):
        self.rng = np.random.default_rng(seed)

    def select_actions(self, states):
        raise NotImplementedError

    def _activate_largest_scores(self, scores):
        """Activate exactly budget arms, using random noise only to break ties."""
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


def _get_model_groups(environment):
    """
    Group arms that share the same P,R model.

    The heterogeneous simulator stores every arm explicitly, but LP-based
    policies only need one LP block per distinct model. This keeps repeated
    arm types cheap while still supporting the fully heterogeneous case.
    """
    if hasattr(environment, "_model_groups_cache"):
        return environment._model_groups_cache

    groups_by_hash = {}
    for arm_index, arm in enumerate(environment.arms):
        model_key = arm.hashname()
        if model_key not in groups_by_hash:
            groups_by_hash[model_key] = {
                "model_key": model_key,
                "arm": arm,
                "indices": [],
            }
        groups_by_hash[model_key]["indices"].append(arm_index)

    groups = list(groups_by_hash.values())
    arm_to_group = np.empty(environment.N, dtype=int)
    for group_id, group in enumerate(groups):
        indices = np.asarray(group["indices"], dtype=int)
        group["group_id"] = group_id
        group["indices"] = indices
        group["weight"] = len(indices) / environment.N
        arm_to_group[indices] = group_id

    environment._model_groups_cache = (groups, arm_to_group)
    return groups, arm_to_group


def _solve_pulp_problem(prob):
    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"LP solver status: {pulp.LpStatus[status]}")


def solve_heterogeneous_average_lp(environment):
    """
    Solve the average-reward relaxed LP for a heterogeneous collection of arms.

    For each distinct model k, Y_k is normalized within that model population:

        sum_s,a Y_k[s,a] = 1.

    The global budget and reward are weighted by the fraction of arms of each
    model. The resource dual is then shared across all model types, which lets
    LP-Priority compare states from different arm models on one scale.
    """
    groups, _ = _get_model_groups(environment)
    actions = range(2)

    prob = pulp.LpProblem("heterogeneous_average_lp", pulp.LpMaximize)
    variables = []
    for group_id, group in enumerate(groups):
        states = range(group["arm"].S)
        variables.append(
            pulp.LpVariable.dicts(
                f"Y_{group_id}",
                (states, actions),
                lowBound=0.0,
            )
        )

    prob += (
        pulp.lpSum(
            group["weight"] * variables[group_id][s][1]
            for group_id, group in enumerate(groups)
            for s in range(group["arm"].S)
        )
        == environment.alpha
    ), "resource"

    for group_id, group in enumerate(groups):
        arm = group["arm"]
        states = range(arm.S)

        for s in states:
            prob += (
                pulp.lpSum(variables[group_id][s][a] for a in actions)
                == pulp.lpSum(
                    variables[group_id][ss][a] * arm.P[ss, a, s]
                    for ss in states
                    for a in actions
                )
            ), f"flow_{group_id}_{s}"

        prob += (
            pulp.lpSum(
                variables[group_id][s][a]
                for s in states
                for a in actions
            )
            == 1.0
        ), f"mass_{group_id}"

    prob += pulp.lpSum(
        group["weight"] * variables[group_id][s][a] * group["arm"].R[s, a]
        for group_id, group in enumerate(groups)
        for s in range(group["arm"].S)
        for a in actions
    )

    _solve_pulp_problem(prob)

    y_by_group = []
    for group_id, group in enumerate(groups):
        arm = group["arm"]
        y = np.zeros((arm.S, arm.A))
        for s in range(arm.S):
            for a in actions:
                y[s, a] = variables[group_id][s][a].varValue
        y_by_group.append(y)

    return {
        "gain": float(pulp.value(prob.objective)),
        "y_by_group": y_by_group,
        "multiplier_alpha": float(prob.constraints["resource"].pi),
        "groups": groups,
    }


def _state_distributions_by_group(environment, states, groups=None):
    if groups is None:
        groups, _ = _get_model_groups(environment)

    x_by_group = []
    for group in groups:
        group_states = np.asarray(states, dtype=int)[group["indices"]]
        counts = np.bincount(group_states, minlength=group["arm"].S)
        x_by_group.append(counts / len(group["indices"]))
    return x_by_group


def _state_count_key_by_group(environment, states, groups=None):
    if groups is None:
        groups, _ = _get_model_groups(environment)

    key = []
    states = np.asarray(states, dtype=int)
    for group in groups:
        counts = np.bincount(
            states[group["indices"]],
            minlength=group["arm"].S,
        )
        key.append(tuple(int(v) for v in counts))
    return tuple(key)


def solve_heterogeneous_finite_lp(environment, x_by_group, time_horizon):
    """
    Solve the finite-horizon LP used by heterogeneous LP-Update.

    This is the direct heterogeneous analogue of
    BanditInstance.relaxed_lp_finite_time(): every model type has its own state
    flow constraints, while all model types share the same resource budget at
    each time step.
    """
    groups, _ = _get_model_groups(environment)
    actions = range(2)
    times = range(time_horizon)

    prob = pulp.LpProblem("heterogeneous_finite_lp", pulp.LpMaximize)
    variables = []
    for group_id, group in enumerate(groups):
        states = range(group["arm"].S)
        variables.append(
            pulp.LpVariable.dicts(
                f"Y_{group_id}",
                (times, states, actions),
                lowBound=0.0,
            )
        )

    for t in times:
        prob += (
            pulp.lpSum(
                group["weight"] * variables[group_id][t][s][1]
                for group_id, group in enumerate(groups)
                for s in range(group["arm"].S)
            )
            == environment.alpha
        ), f"resource_{t}"

    for group_id, group in enumerate(groups):
        arm = group["arm"]
        states = range(arm.S)

        for s in states:
            prob += (
                pulp.lpSum(variables[group_id][0][s][a] for a in actions)
                == float(x_by_group[group_id][s])
            ), f"initial_{group_id}_{s}"

        for t in range(time_horizon - 1):
            for s in states:
                prob += (
                    pulp.lpSum(
                        variables[group_id][t + 1][s][a]
                        for a in actions
                    )
                    == pulp.lpSum(
                        variables[group_id][t][ss][a] * arm.P[ss, a, s]
                        for ss in states
                        for a in actions
                    )
                ), f"flow_{group_id}_{t}_{s}"

    prob += pulp.lpSum(
        group["weight"]
        * variables[group_id][t][s][a]
        * group["arm"].R[s, a]
        for group_id, group in enumerate(groups)
        for t in times
        for s in range(group["arm"].S)
        for a in actions
    )

    _solve_pulp_problem(prob)

    y_by_group = []
    for group_id, group in enumerate(groups):
        arm = group["arm"]
        y = np.zeros((time_horizon, arm.S, arm.A))
        for t in times:
            for s in range(arm.S):
                for a in actions:
                    y[t, s, a] = variables[group_id][t][s][a].varValue
        y_by_group.append(y)

    return {
        "gain": float(pulp.value(prob.objective)),
        "y_by_group": y_by_group,
        "groups": groups,
    }


def _activation_probabilities_from_y(y_by_group):
    probabilities = []
    for y in y_by_group:
        x = np.sum(y, axis=1)
        probabilities.append(
            np.divide(
                y[:, 1],
                x,
                out=np.zeros_like(x),
                where=x > 1e-12,
            )
        )
    return probabilities


def _repair_binary_actions_to_budget(actions, budget, rng):
    """Randomly repair a binary action vector to satisfy the exact budget."""
    actions = np.asarray(actions, dtype=int).copy()
    current = int(np.sum(actions))

    if current > budget:
        active = np.flatnonzero(actions == 1)
        turn_off = rng.choice(active, size=current - budget, replace=False)
        actions[turn_off] = 0
    elif current < budget:
        passive = np.flatnonzero(actions == 0)
        turn_on = rng.choice(passive, size=budget - current, replace=False)
        actions[turn_on] = 1

    return actions


class HeterogeneousWhittlePolicy(HeterogeneousPolicy):
    """
    Compute an index table for each distinct arm model.

    Arms with identical P_i and R_i share one index computation. In a fully
    heterogeneous experiment, every arm may require its own index table.
    """

    def __init__(self, environment):
        super().__init__(environment)
        index_by_model = {}
        self.indices_by_arm = []

        for arm in environment.arms:
            model_key = arm.hashname()
            if model_key not in index_by_model:
                teacher_policy = strategies.WhittleIndexStrategy(
                    arm,
                    environment.alpha,
                )
                index_by_model[model_key] = np.asarray(
                    teacher_policy.whittle_indices,
                    dtype=float,
                )
            self.indices_by_arm.append(index_by_model[model_key])

        self.num_distinct_index_models = len(index_by_model)

    def select_actions(self, states):
        scores = np.array(
            [
                self.indices_by_arm[i][states[i]]
                for i in range(self.N)
            ]
        )
        return self._activate_largest_scores(scores)


class HeterogeneousLPPriorityPolicy(HeterogeneousPolicy):
    """
    LP-priority for heterogeneous arms.

    First solve one global heterogeneous relaxed LP to obtain a common resource
    multiplier. Then compute one state-priority table for each distinct arm
    model using that shared multiplier.
    """

    def __init__(self, environment):
        super().__init__(environment)
        self.groups, self.arm_to_group = _get_model_groups(environment)
        lp_solution = solve_heterogeneous_average_lp(environment)
        self.multiplier_alpha = lp_solution["multiplier_alpha"]
        self.lp_indices_by_group = []

        for group in self.groups:
            arm = group["arm"]
            q_table = np.copy(arm.R)
            reward_plus_penalty = np.copy(arm.R)
            reward_plus_penalty[:, 1] -= self.multiplier_alpha

            for _ in range(1000):
                q_table = reward_plus_penalty + arm.P @ np.max(q_table, axis=1)

            self.lp_indices_by_group.append(q_table[:, 1] - q_table[:, 0])

    def select_actions(self, states):
        scores = np.array(
            [
                self.lp_indices_by_group[self.arm_to_group[i]][states[i]]
                for i in range(self.N)
            ]
        )
        return self._activate_largest_scores(scores)


class HeterogeneousFTVAPolicy(HeterogeneousPolicy):
    """
    Heterogeneous FTVA using per-model relaxed-LP activation probabilities.

    The real arms are simulated by simulate_heterogeneous(). This policy keeps a
    virtual copy of every arm state and updates it after observing the real
    transition.
    """

    def __init__(self, environment):
        super().__init__(environment)
        self.groups, self.arm_to_group = _get_model_groups(environment)
        lp_solution = solve_heterogeneous_average_lp(environment)
        self.pi_by_group = _activation_probabilities_from_y(
            lp_solution["y_by_group"]
        )
        self.virtual_states = None
        self.last_virtual_actions = None
        self.last_virtual_states = None

    def reset(self, seed):
        super().reset(seed)
        self.virtual_states = None
        self.last_virtual_actions = None
        self.last_virtual_states = None

    def select_actions(self, states):
        states = np.asarray(states, dtype=int)
        if self.virtual_states is None or len(self.virtual_states) != self.N:
            self.virtual_states = states.copy()

        self.last_virtual_states = self.virtual_states.copy()
        probabilities = np.array(
            [
                self.pi_by_group[self.arm_to_group[i]][self.virtual_states[i]]
                for i in range(self.N)
            ]
        )
        virtual_actions = (
            self.rng.random(self.N) <= probabilities
        ).astype(int)
        actions = _repair_binary_actions_to_budget(
            virtual_actions,
            self.budget,
            self.rng,
        )

        self.last_virtual_actions = virtual_actions
        return actions

    def observe_transition(self, states, actions, rewards, next_states):
        del rewards
        states = np.asarray(states, dtype=int)
        actions = np.asarray(actions, dtype=int)
        next_states = np.asarray(next_states, dtype=int)

        new_virtual_states = np.empty(self.N, dtype=int)
        for i in range(self.N):
            group_id = self.arm_to_group[i]
            arm = self.groups[group_id]["arm"]
            virtual_state = self.last_virtual_states[i]
            virtual_action = self.last_virtual_actions[i]

            if states[i] == virtual_state and actions[i] == virtual_action:
                # Coupled update: when the virtual and real arms agree, keep
                # them together by reusing the observed real transition.
                new_virtual_states[i] = next_states[i]
            else:
                new_virtual_states[i] = self.rng.choice(
                    arm.S,
                    p=arm.P[virtual_state, virtual_action, :],
                )

        self.virtual_states = new_virtual_states


class HeterogeneousLPUpdatePolicy(HeterogeneousPolicy):
    """
    Heterogeneous LP-Update / MPC.

    At each decision time, build the empirical state distribution for every
    distinct arm model, solve a finite-horizon heterogeneous LP, and use its
    first-stage activation probabilities as scores for the individual arms.
    """

    def __init__(self, environment, time_horizon=20):
        super().__init__(environment)
        self.groups, self.arm_to_group = _get_model_groups(environment)
        self.time_horizon = int(time_horizon)
        self.computed_scores = {}

    def select_actions(self, states):
        states = np.asarray(states, dtype=int)
        key = _state_count_key_by_group(self.environment, states, self.groups)

        if key not in self.computed_scores:
            x_by_group = _state_distributions_by_group(
                self.environment,
                states,
                self.groups,
            )
            lp_solution = solve_heterogeneous_finite_lp(
                self.environment,
                x_by_group=x_by_group,
                time_horizon=self.time_horizon,
            )
            y0_by_group = [
                y_by_time[0]
                for y_by_time in lp_solution["y_by_group"]
            ]
            self.computed_scores[key] = _activation_probabilities_from_y(
                y0_by_group
            )

        probabilities_by_group = self.computed_scores[key]
        scores = np.array(
            [
                probabilities_by_group[self.arm_to_group[i]][states[i]]
                for i in range(self.N)
            ]
        )
        return self._activate_largest_scores(scores)


class HeterogeneousMyopicPolicy(HeterogeneousPolicy):
    """Rank arms by their current one-step active reward advantage."""

    def select_actions(self, states):
        scores = np.array(
            [
                arm.R[states[i], 1] - arm.R[states[i], 0]
                for i, arm in enumerate(self.environment.arms)
            ]
        )
        return self._activate_largest_scores(scores)


class HeterogeneousRandomActivationPolicy(HeterogeneousPolicy):
    """Uniformly choose budget distinct arms at every decision time."""

    def select_actions(self, states):
        del states
        actions = np.zeros(self.N, dtype=int)
        if self.budget > 0:
            selected = self.rng.choice(
                self.N,
                size=self.budget,
                replace=False,
            )
            actions[selected] = 1
        return actions


class HeterogeneousRoundRobinPolicy(HeterogeneousPolicy):
    """Cycle through arm identities without looking at states, P_i, or R_i."""

    def __init__(self, environment):
        super().__init__(environment)
        self.pointer = 0

    def reset(self, seed):
        super().reset(seed)
        self.pointer = 0

    def select_actions(self, states):
        del states
        actions = np.zeros(self.N, dtype=int)
        selected = (self.pointer + np.arange(self.budget)) % self.N
        actions[selected] = 1
        self.pointer = (self.pointer + self.budget) % self.N
        return actions


POLICY_CLASSES = {
    "HeterogeneousWhittle": HeterogeneousWhittlePolicy,
    "HeterogeneousLPPriority": HeterogeneousLPPriorityPolicy,
    "HeterogeneousFTVA": HeterogeneousFTVAPolicy,
    "HeterogeneousLPUpdate": HeterogeneousLPUpdatePolicy,
    "HeterogeneousMyopic": HeterogeneousMyopicPolicy,
    "RandomActivation": HeterogeneousRandomActivationPolicy,
    "RoundRobin": HeterogeneousRoundRobinPolicy,
}


def make_heterogeneous_policy(policy_name, environment):
    """Create one policy under the common explicit-arm interface."""
    if policy_name not in POLICY_CLASSES:
        raise ValueError(f"Unknown heterogeneous policy: {policy_name}")
    return POLICY_CLASSES[policy_name](environment)


def simulate_heterogeneous(environment, policy, horizon, seed):
    """
    Simulate all arms explicitly and return per-arm average reward.

    Reward is normalized by N, matching the scale used by the homogeneous
    benchmark.
    """
    environment_rng = np.random.default_rng(seed)
    policy.reset(seed + 1)

    states = np.array(
        [
            environment_rng.choice(arm.S, p=initial)
            for arm, initial in zip(
                environment.arms,
                environment.initial_distributions,
            )
        ],
        dtype=int,
    )
    reward_history = np.zeros(horizon)

    for t in range(horizon):
        actions = np.asarray(policy.select_actions(states), dtype=int)
        if actions.shape != (environment.N,):
            raise ValueError("Policy returned an action vector with wrong shape.")
        if np.sum(actions) != environment.budget:
            raise ValueError("Policy did not satisfy the exact activation budget.")

        rewards = np.empty(environment.N)
        next_states = np.empty(environment.N, dtype=int)

        for i, arm in enumerate(environment.arms):
            state = states[i]
            action = actions[i]
            rewards[i] = arm.R[state, action]
            next_states[i] = environment_rng.choice(
                arm.S,
                p=arm.P[state, action, :],
            )

        reward_history[t] = np.mean(rewards)
        if hasattr(policy, "observe_transition"):
            policy.observe_transition(states, actions, rewards, next_states)
        states = next_states

    return {
        "mean_reward": float(np.mean(reward_history)),
        "reward_history": reward_history,
        "final_states": states,
    }


def _resolve_type_count(N, num_types):
    """None means every arm receives its own parameter values."""
    if num_types is None:
        return N
    if not 1 <= num_types <= N:
        raise ValueError("num_types must be between 1 and N, or None.")
    return int(num_types)


def build_heterogeneous_maintenance(
    bandit_lp,
    N,
    alpha=0.2,
    state_count=5,
    num_types=5,
):
    """
    Build maintenance arms with different degradation and repair parameters.

    Set num_types=5 for five recurring machine types.
    Set num_types=None to give every machine a distinct P_i and R_i.
    """
    type_count = _resolve_type_count(N, num_types)
    arms_by_type = []
    initial_by_type = []
    parameters_by_type = []

    for type_id in range(type_count):
        fraction = type_id / max(type_count - 1, 1)
        parameters = {
            "p_degrade": 0.16 + 0.18 * fraction,
            "p_big_degrade": 0.03 + 0.04 * fraction,
            "repair_success": 0.93 - 0.18 * fraction,
            "repair_cost": 0.08 + 0.16 * fraction,
        }
        spec = maintenance_degradation_instance(
            bandit_lp=bandit_lp,
            state_count=state_count,
            alpha=alpha,
            **parameters,
        )
        arms_by_type.append(spec.bandit)
        initial_by_type.append(spec.initial_state)
        parameters_by_type.append(parameters)

    type_ids = np.arange(N) % type_count
    return HeterogeneousRMAB(
        name=(
            f"heterogeneous_maintenance_S{state_count}_"
            f"{type_count}types"
        ),
        arms=[arms_by_type[type_id] for type_id in type_ids],
        alpha=alpha,
        initial_distributions=[
            initial_by_type[type_id] for type_id in type_ids
        ],
        type_ids=type_ids,
        arm_parameters=[
            parameters_by_type[type_id] for type_id in type_ids
        ],
        source="Synthetic heterogeneous extension of maintenance_degradation_instance",
    )


def build_heterogeneous_wireless(
    bandit_lp,
    N,
    alpha=0.4,
    state_count=5,
    num_types=5,
):
    """
    Build wireless arms with different channel transition parameters.

    Set num_types=5 for five recurring channel types.
    Set num_types=None to give every channel a distinct P_i and R_i.
    """
    type_count = _resolve_type_count(N, num_types)
    arms_by_type = []
    initial_by_type = []
    parameters_by_type = []

    for type_id in range(type_count):
        fraction = type_id / max(type_count - 1, 1)
        parameters = {
            "p_up": 0.14 + 0.12 * fraction,
            "p_down": 0.08 + 0.08 * fraction,
            "active_extra_down": 0.01 + 0.05 * fraction,
        }
        spec = wireless_channel_instance(
            bandit_lp=bandit_lp,
            state_count=state_count,
            alpha=alpha,
            **parameters,
        )
        arms_by_type.append(spec.bandit)
        initial_by_type.append(spec.initial_state)
        parameters_by_type.append(parameters)

    type_ids = np.arange(N) % type_count
    return HeterogeneousRMAB(
        name=(
            f"heterogeneous_wireless_S{state_count}_"
            f"{type_count}types"
        ),
        arms=[arms_by_type[type_id] for type_id in type_ids],
        alpha=alpha,
        initial_distributions=[
            initial_by_type[type_id] for type_id in type_ids
        ],
        type_ids=type_ids,
        arm_parameters=[
            parameters_by_type[type_id] for type_id in type_ids
        ],
        source="Synthetic heterogeneous extension of wireless_channel_instance",
    )
