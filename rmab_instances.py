"""
Instance library for RMAB benchmark experiments.

All instances in this file are converted into the teacher's BanditInstance(P, R)
format, where:

    P.shape == (S, 2, S)
    R.shape == (S, 2)

This means they can be compared with the current model-known policies:
Whittle, LP-Priority, FTVA, LP-Update, Myopic, RandomPriority, LPRandomized.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class RMABInstanceSpec:
    name: str
    bandit: object
    default_alpha: float
    initial_state: np.ndarray
    source: str
    notes: str = ""


def uniform_initial_state(number_of_states):
    return np.ones(number_of_states) / number_of_states


def yan3_instance(bandit_lp):
    """
    Third 3-state counterexample from the public Hong et al. codebase.

    Source code location:
    YigeHong/rb-break-ugap-ftva, rb_settings.py, class Gast20Example3.
    """
    P0 = np.array(
        [
            [0.47819592, 0.02090623, 0.50089785],
            [0.08063373, 0.15456935, 0.76479692],
            [0.66552514, 0.08481946, 0.24965540],
        ]
    )
    P1 = np.array(
        [
            [0.00279465, 0.37327924, 0.62392611],
            [0.51582335, 0.46333908, 0.02083756],
            [0.41875202, 0.17776712, 0.40348086],
        ]
    )
    R0 = np.zeros(3)
    R1 = np.array([0.97658608, 0.53014109, 0.40394919])

    P = np.zeros((3, 2, 3))
    R = np.zeros((3, 2))
    P[:, 0, :] = P0
    P[:, 1, :] = P1
    R[:, 0] = R0
    R[:, 1] = R1

    bandit = bandit_lp.BanditInstance(P, R)
    return RMABInstanceSpec(
        name="yan_gast_example3",
        bandit=bandit,
        default_alpha=0.4,
        initial_state=uniform_initial_state(3),
        source="YigeHong/rb-break-ugap-ftva: Gast20Example3",
        notes="3-state cycling-style example.",
    )


def conveyor_instance(bandit_lp, name="eg4unif-tb", state_count=8):
    """
    Conveyor counterexamples from the public Hong et al. codebase.

    Source code location:
    YigeHong/rb-break-ugap-ftva, rb_settings.py, class ConveyorExample.
    """
    assert state_count % 2 == 0, "Conveyor state_count must be even."
    half_size = state_count // 2

    if name == "eg4unif-tb":
        probs_R = np.ones(state_count) / 6.0
        probs_L = np.ones(state_count) * 0.9
        probs_L[0] = 0.0
        action_script = np.zeros(state_count, dtype=int)
        action_script[:half_size] = 1
    elif name == "eg4action-gap-tb":
        probs_R = np.ones(state_count) * 0.1
        probs_L = 0.5 - 0.01 * np.arange(state_count)
        probs_L[1] = 1.0
        probs_L[0] = 0.0
        action_script = np.zeros(state_count, dtype=int)
        action_script[:half_size] = 1
    else:
        raise ValueError(f"Unknown conveyor instance: {name}")

    P = np.zeros((state_count, 2, state_count))
    R = np.zeros((state_count, 2))

    for s in range(state_count):
        action_R = action_script[s]
        action_L = 1 - action_R

        if s == 0:
            P[s, action_R, s + 1] = probs_R[s]
            P[s, action_R, s] = 1 - probs_R[s]
            P[s, action_L, s + 1] = probs_L[s]
            P[s, action_L, s] = 1 - probs_L[s]
        elif s == state_count - 1:
            P[s, action_R, 0] = probs_R[s]
            P[s, action_R, s] = 1 - probs_R[s]
            P[s, action_L, s - 1] = probs_L[s]
            P[s, action_L, s] = 1 - probs_L[s]
        else:
            P[s, action_R, s + 1] = probs_R[s]
            P[s, action_R, s] = 1 - probs_R[s]
            P[s, action_L, s - 1] = probs_L[s]
            P[s, action_L, s] = 1 - probs_L[s]

    last_state = state_count - 1
    R[last_state, action_script[last_state]] = probs_R[last_state]

    tries_in_a_loop = 1.0 / probs_R
    suggested_alpha = float(
        np.sum(tries_in_a_loop * action_script) / np.sum(tries_in_a_loop)
    )

    bandit = bandit_lp.BanditInstance(P, R)
    return RMABInstanceSpec(
        name=f"conveyor_{name}_S{state_count}",
        bandit=bandit,
        default_alpha=suggested_alpha,
        initial_state=uniform_initial_state(state_count),
        source=f"YigeHong/rb-break-ugap-ftva: ConveyorExample {name}",
        notes="Structured conveyor example designed to challenge priority rules.",
    )


def recovering_bandit_instance(
    bandit_lp,
    max_wait=20,
    theta=(10.0, 0.2, 0.0),
    alpha=0.2,
):
    """
    Finite-state recovering bandit.

    State s means the arm has waited s+1 periods. Passive action increases
    waiting time up to max_wait. Active action gives a recovery reward and
    resets the arm to wait=1.

    This is adapted from the Recovering Bandits environment used in NeurWIN.
    """
    theta0, theta1, theta2 = theta
    S = max_wait
    P = np.zeros((S, 2, S))
    R = np.zeros((S, 2))

    for s in range(S):
        wait = s + 1

        # action 0: passive, wait one more step
        next_passive = min(s + 1, S - 1)
        P[s, 0, next_passive] = 1.0
        R[s, 0] = 0.0

        # action 1: active, collect reward and reset
        P[s, 1, 0] = 1.0
        R[s, 1] = theta0 * (1.0 - np.exp(-theta1 * wait + theta2))

    bandit = bandit_lp.BanditInstance(P, R)
    return RMABInstanceSpec(
        name=f"recovering_maxwait{max_wait}",
        bandit=bandit,
        default_alpha=alpha,
        initial_state=uniform_initial_state(S),
        source="Recovering Bandits / NeurWIN-style finite truncation",
        notes="Deterministic passive aging and active reset.",
    )


def random_instance(bandit_lp, state_count=10, action_count=2, seed=123, alpha=0.4):
    bandit = bandit_lp.BanditRandom(
        number_of_states=state_count,
        number_of_actions=action_count,
        seed=seed,
    )
    return RMABInstanceSpec(
        name=f"random_S{state_count}_seed{seed}",
        bandit=bandit,
        default_alpha=alpha,
        initial_state=uniform_initial_state(state_count),
        source="Synthetic dense random BanditRandom instance",
        notes="Random transition and reward matrices from teacher code.",
    )


def teacher_hong_counterexample(bandit_lp):
    bandit = bandit_lp.BanditCounterExample()
    return RMABInstanceSpec(
        name="hong_counterexample",
        bandit=bandit,
        default_alpha=0.5,
        initial_state=uniform_initial_state(bandit.S),
        source="Teacher code: BanditCounterExample",
        notes="8-state counterexample used in the paper notebook.",
    )


def teacher_yan1(bandit_lp):
    bandit = bandit_lp.BanditCounterExampleYan1()
    return RMABInstanceSpec(
        name="yan_gast_example1",
        bandit=bandit,
        default_alpha=0.4,
        initial_state=uniform_initial_state(bandit.S),
        source="Teacher code: BanditCounterExampleYan1",
        notes="3-state example from Gast/Yan-style counterexamples.",
    )


def teacher_yan2(bandit_lp):
    bandit = bandit_lp.BanditCounterExampleYan2()
    return RMABInstanceSpec(
        name="yan_gast_example2",
        bandit=bandit,
        default_alpha=0.4,
        initial_state=uniform_initial_state(bandit.S),
        source="Teacher code: BanditCounterExampleYan2",
        notes="3-state example from Gast/Yan-style counterexamples.",
    )


def build_instance_library(bandit_lp):
    """
    Return a dictionary of all currently supported model-known instances.
    """
    specs = [
        random_instance(bandit_lp, state_count=10, seed=123, alpha=0.4),
        teacher_hong_counterexample(bandit_lp),
        teacher_yan1(bandit_lp),
        teacher_yan2(bandit_lp),
        yan3_instance(bandit_lp),
        recovering_bandit_instance(bandit_lp, max_wait=20, alpha=0.2),
    ]
    return {spec.name: spec for spec in specs}


def instance_metadata_dataframe(instance_library):
    """
    Convert the instance library into a small dataframe-friendly list of dicts.
    """
    rows = []
    for spec in instance_library.values():
        rows.append(
            {
                "instance": spec.name,
                "S": spec.bandit.S,
                "A": spec.bandit.A,
                "default_alpha": spec.default_alpha,
                "source": spec.source,
                "notes": spec.notes,
            }
        )
    return rows
