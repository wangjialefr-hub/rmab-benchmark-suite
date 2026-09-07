"""
Extra known-model RMAB instances.

This file only builds RMAB instances in the teacher code format:

    P.shape == (S, 2, S)
    R.shape == (S, 2)

All instances here assume the model is known: algorithms are allowed to use the
true transition matrix P and reward matrix R.

The main entry points are:

    build_extra_known_model_instance_library(bandit_lp)
    build_extended_known_model_instance_library(bandit_lp)
    extra_instance_metadata_dataframe(instance_library)
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class ExtraRMABInstanceSpec:
    name: str
    bandit: object
    default_alpha: float
    initial_state: np.ndarray
    source: str
    notes: str
    group: str
    transition_summary: str
    reward_summary: str


def uniform_initial_state(number_of_states):
    return np.ones(number_of_states) / number_of_states


def alpha_tag(alpha):
    return f"a{int(round(100 * alpha)):02d}"


def validate_matrices(P, R):
    if P.ndim != 3:
        raise ValueError("P must have shape (S, A, S).")
    if R.ndim != 2:
        raise ValueError("R must have shape (S, A).")
    S, A = R.shape
    if P.shape != (S, A, S):
        raise ValueError(f"P shape {P.shape} is incompatible with R shape {R.shape}.")
    if A != 2:
        raise ValueError("Current teacher policies assume A=2.")
    if np.any(P < -1e-12):
        raise ValueError("P contains negative probabilities.")
    row_sums = P.sum(axis=2)
    if not np.allclose(row_sums, 1.0, atol=1e-10):
        raise ValueError("Every P[s, a, :] row must sum to 1.")


def make_spec(
    bandit_lp,
    name,
    P,
    R,
    alpha,
    source,
    notes,
    group,
    transition_summary,
    reward_summary,
    initial_state=None,
):
    validate_matrices(P, R)
    bandit = bandit_lp.BanditInstance(P.astype(float), R.astype(float))
    if initial_state is None:
        initial_state = uniform_initial_state(bandit.S)
    return ExtraRMABInstanceSpec(
        name=name,
        bandit=bandit,
        default_alpha=float(alpha),
        initial_state=np.asarray(initial_state, dtype=float),
        source=source,
        notes=notes,
        group=group,
        transition_summary=transition_summary,
        reward_summary=reward_summary,
    )


# ============================================================
# 1. Random robustness family
# ============================================================


def random_family_instances(
    bandit_lp,
    state_counts=(5, 10, 20),
    seeds=(101, 202, 303),
    alphas=(0.2, 0.4, 0.6),
):
    """
    Synthetic random instances for robustness checks.

    For each S and seed, the teacher's BanditRandom generates dense random P
    and R. We reuse the same P,R under several alpha values to separate model
    randomness from budget effects.
    """
    specs = []
    for S in state_counts:
        for seed in seeds:
            bandit = bandit_lp.BanditRandom(
                number_of_states=S,
                number_of_actions=2,
                seed=seed,
            )
            for alpha in alphas:
                specs.append(
                    ExtraRMABInstanceSpec(
                        name=f"random_S{S}_seed{seed}_{alpha_tag(alpha)}",
                        bandit=bandit,
                        default_alpha=float(alpha),
                        initial_state=uniform_initial_state(S),
                        source="Synthetic robustness family using teacher BanditRandom",
                        notes="Dense random transition matrix and dense random rewards.",
                        group="random",
                        transition_summary=(
                            "P[s,a,:] is sampled from independent exponential entries "
                            "and normalized to a probability vector."
                        ),
                        reward_summary=(
                            "R[s,a] is sampled from independent exponential entries."
                        ),
                    )
                )
    return specs


# ============================================================
# 2. Hong / Gast / FTVA-style counterexamples
# ============================================================


GAST20_EXAMPLES = {
    1: {
        "P0": [
            [0.52140730, 0.40392496, 0.07466774],
            [0.01584150, 0.21455666, 0.76960184],
            [0.53722329, 0.37651148, 0.08626522],
        ],
        "P1": [
            [0.24639364, 0.23402385, 0.51958251],
            [0.49681581, 0.49509821, 0.00808597],
            [0.37826553, 0.15469252, 0.46704195],
        ],
        "R1": [0.72232506, 0.18844869, 0.25752477],
    },
    2: {
        "P0": [
            [0.02232142, 0.10229283, 0.87538575],
            [0.03426605, 0.17175704, 0.79397691],
            [0.52324756, 0.45523298, 0.02151947],
        ],
        "P1": [
            [0.14874601, 0.30435809, 0.54689589],
            [0.56845754, 0.41117331, 0.02036915],
            [0.25265570, 0.27310439, 0.47423990],
        ],
        "R1": [0.37401552, 0.11740814, 0.07866135],
    },
    3: {
        "P0": [
            [0.47819592, 0.02090623, 0.50089785],
            [0.08063373, 0.15456935, 0.76479692],
            [0.66552514, 0.08481946, 0.24965540],
        ],
        "P1": [
            [0.00279465, 0.37327924, 0.62392611],
            [0.51582335, 0.46333908, 0.02083756],
            [0.41875202, 0.17776712, 0.40348086],
        ],
        "R1": [0.97658608, 0.53014109, 0.40394919],
    },
}


def gast20_instance(bandit_lp, example_id):
    data = GAST20_EXAMPLES[example_id]
    P = np.zeros((3, 2, 3))
    R = np.zeros((3, 2))
    P[:, 0, :] = np.array(data["P0"], dtype=float)
    P[:, 1, :] = np.array(data["P1"], dtype=float)
    R[:, 1] = np.array(data["R1"], dtype=float)
    return make_spec(
        bandit_lp=bandit_lp,
        name=f"gast20_example{example_id}",
        P=P,
        R=R,
        alpha=0.4,
        source=(
            "YigeHong/rb-break-ugap-ftva rb_settings.py, "
            f"Gast20Example{example_id}; related to Gast et al. finite-N examples"
        ),
        notes="Small 3-state counterexample family used to stress finite-N behavior.",
        group="hong_gast_ftva",
        transition_summary="S=3. Both actions use dense 3-by-3 transition matrices P0 and P1.",
        reward_summary="Passive rewards are zero. Active rewards R[:,1] are state dependent.",
    )


def conveyor_parameters(name, state_count):
    if state_count % 2 != 0:
        raise ValueError("Conveyor state_count must be even.")
    half_size = state_count // 2

    if name == "eg4unif-tb":
        probs_R = np.ones(state_count) / 6.0
        probs_L = np.ones(state_count) * 0.9
        probs_L[0] = 0.0
    elif name == "eg4action-gap-tb":
        probs_R = np.ones(state_count) * 0.1
        probs_L = 0.5 - 0.01 * np.arange(state_count)
        probs_L[1] = 1.0
        probs_L[0] = 0.0
    elif name == "eg4archive1":
        probs_R = np.ones(state_count) / 6.0
        probs_L = np.ones(state_count) * 0.9
        probs_L[0] = 1.0
    elif name == "eg4archive2":
        probs_R = np.ones(state_count) / 6.0
        probs_L = np.ones(state_count) / (2 + 0.1 * state_count)
        probs_L[1] = 0.5
        probs_L[0] = 1.0 / 3.0
    else:
        raise ValueError(f"Unknown conveyor parameter set: {name}")

    action_script = np.zeros(state_count, dtype=int)
    action_script[:half_size] = 1
    tries_in_a_loop = 1.0 / probs_R
    suggested_alpha = float(np.sum(tries_in_a_loop * action_script) / np.sum(tries_in_a_loop))
    return probs_L, probs_R, action_script, suggested_alpha


def conveyor_instance(bandit_lp, name, state_count=8):
    probs_L, probs_R, action_script, suggested_alpha = conveyor_parameters(name, state_count)

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

    return make_spec(
        bandit_lp=bandit_lp,
        name=f"conveyor_{name}_S{state_count}",
        P=P,
        R=R,
        alpha=suggested_alpha,
        source=f"YigeHong/rb-break-ugap-ftva rb_settings.py, ConveyorExample {name}",
        notes="Conveyor counterexample/stress-test family for priority and tie-breaking rules.",
        group="hong_gast_ftva",
        transition_summary=(
            f"S={state_count}. Correct action moves right with probability probs_R[s]; "
            "wrong action moves left with probability probs_L[s]; right boundary wraps to state 0."
        ),
        reward_summary=(
            "Reward is zero except at the rightmost state, where taking the correct action "
            "earns probs_R[last_state]."
        ),
    )


def hong_gast_ftva_instances(bandit_lp):
    specs = [gast20_instance(bandit_lp, example_id) for example_id in (1, 2, 3)]
    # Keep non-duplicate conveyor examples. conveyor_eg4action-gap-tb_S8 is
    # excluded because it has the same P,R,alpha as hong_counterexample.
    for name in ("eg4unif-tb", "eg4archive1", "eg4archive2"):
        specs.append(conveyor_instance(bandit_lp, name=name, state_count=8))
    return specs


# ============================================================
# 3. Machine maintenance / degradation
# ============================================================


def maintenance_degradation_instance(
    bandit_lp,
    state_count=5,
    alpha=0.3,
    p_degrade=0.25,
    p_big_degrade=0.05,
    repair_success=0.85,
    repair_cost=0.15,
):
    """
    State s is machine degradation level: 0 is healthy, S-1 is failed.

    Passive action: machine tends to deteriorate.
    Active action: maintenance tends to repair/reset the machine.
    """
    S = state_count
    P = np.zeros((S, 2, S))
    R = np.zeros((S, 2))

    for s in range(S):
        # action 0: no maintenance
        stay_prob = max(0.0, 1.0 - p_degrade - p_big_degrade)
        P[s, 0, s] += stay_prob
        P[s, 0, min(s + 1, S - 1)] += p_degrade
        P[s, 0, min(s + 2, S - 1)] += p_big_degrade

        # action 1: maintenance
        improved_state = max(s - 1, 0)
        P[s, 1, 0] += repair_success
        P[s, 1, improved_state] += 1.0 - repair_success

        health = 1.0 - s / max(S - 1, 1)
        urgency_bonus = s / max(S - 1, 1)
        R[s, 0] = health
        R[s, 1] = 1.0 - repair_cost + 0.15 * urgency_bonus

    return make_spec(
        bandit_lp=bandit_lp,
        name=f"maintenance_S{S}_{alpha_tag(alpha)}",
        P=P,
        R=R,
        alpha=alpha,
        source="Synthetic condition-based maintenance/degradation RMAB instance",
        notes="Structured maintenance model: passive degradation and active repair.",
        group="maintenance",
        transition_summary=(
            f"S={S}. State is degradation level. Passive action moves to worse states; "
            "active action repairs to state 0 with high probability."
        ),
        reward_summary=(
            "Passive reward is current machine health. Active reward is repaired-operation "
            "value minus maintenance cost, with a small bonus for severe degradation."
        ),
    )


def maintenance_family_instances(bandit_lp):
    specs = []
    for S in (5, 10):
        for alpha in (0.2, 0.4):
            specs.append(maintenance_degradation_instance(bandit_lp, state_count=S, alpha=alpha))
    return specs


# ============================================================
# 4. Wireless scheduling / channel state
# ============================================================


def wireless_channel_instance(
    bandit_lp,
    state_count=5,
    alpha=0.4,
    p_up=0.20,
    p_down=0.12,
    active_extra_down=0.03,
):
    """
    State s is channel quality: 0 is bad, S-1 is good.

    Passive action gives no transmission reward. Active action transmits and
    earns a throughput reward increasing with channel quality.
    """
    S = state_count
    P = np.zeros((S, 2, S))
    R = np.zeros((S, 2))

    for s in range(S):
        for a in (0, 1):
            down = min(0.95, p_down + active_extra_down * a)
            up = p_up
            stay = max(0.0, 1.0 - up - down)

            P[s, a, s] += stay
            P[s, a, min(s + 1, S - 1)] += up
            P[s, a, max(s - 1, 0)] += down

        quality = s / max(S - 1, 1)
        R[s, 0] = 0.0
        R[s, 1] = 0.1 + quality ** 1.3

    return make_spec(
        bandit_lp=bandit_lp,
        name=f"wireless_channel_S{S}_{alpha_tag(alpha)}",
        P=P,
        R=R,
        alpha=alpha,
        source="Synthetic wireless scheduling/channel-state RMAB instance",
        notes="Structured scheduling model inspired by channel-state RMAB benchmarks.",
        group="wireless",
        transition_summary=(
            f"S={S}. State is channel quality. Both actions follow a birth-death channel "
            "process; active transmission slightly increases degradation probability."
        ),
        reward_summary=(
            "Passive reward is zero. Active reward is throughput, increasing with channel quality."
        ),
    )


def wireless_family_instances(bandit_lp):
    specs = []
    for S in (5, 10):
        for alpha in (0.2, 0.4):
            specs.append(wireless_channel_instance(bandit_lp, state_count=S, alpha=alpha))
    return specs


# ============================================================
# 5. Deadline scheduling / urgency
# ============================================================


def deadline_scheduling_instance(
    bandit_lp,
    state_count=5,
    alpha=0.3,
    miss_penalty=0.05,
):
    """
    State s is urgency / closeness to deadline.

    Passive action makes the job more urgent. Active action serves the job and
    resets it to the least urgent state.
    """
    S = state_count
    P = np.zeros((S, 2, S))
    R = np.zeros((S, 2))

    for s in range(S):
        urgency = (s + 1) / S

        # action 0: wait, deadline gets closer
        P[s, 0, min(s + 1, S - 1)] = 1.0

        # action 1: serve, new job arrives
        P[s, 1, 0] = 1.0

        R[s, 0] = -miss_penalty * urgency
        R[s, 1] = urgency

    return make_spec(
        bandit_lp=bandit_lp,
        name=f"deadline_S{S}_{alpha_tag(alpha)}",
        P=P,
        R=R,
        alpha=alpha,
        source="Synthetic deadline/urgency scheduling RMAB instance",
        notes="Structured deadline model: passive aging and active service reset.",
        group="deadline",
        transition_summary=(
            f"S={S}. State is urgency. Passive action moves deterministically toward the "
            "deadline; active action serves and resets to state 0."
        ),
        reward_summary=(
            "Passive action has a small urgency penalty. Active reward increases with urgency."
        ),
    )


def deadline_family_instances(bandit_lp):
    specs = []
    for S in (5, 10):
        for alpha in (0.2, 0.4):
            specs.append(deadline_scheduling_instance(bandit_lp, state_count=S, alpha=alpha))
    return specs


# ============================================================
# Public builders
# ============================================================


def build_extra_known_model_instance_specs(bandit_lp):
    """
    Return a list of all extra known-model instance specs.
    """
    specs = []
    specs.extend(random_family_instances(bandit_lp))
    specs.extend(hong_gast_ftva_instances(bandit_lp))
    specs.extend(maintenance_family_instances(bandit_lp))
    specs.extend(wireless_family_instances(bandit_lp))
    specs.extend(deadline_family_instances(bandit_lp))
    return specs


def build_extra_known_model_instance_library(bandit_lp):
    """
    Return a dictionary {instance_name: spec} for the extra instances only.
    """
    specs = build_extra_known_model_instance_specs(bandit_lp)
    return {spec.name: spec for spec in specs}


def build_extended_known_model_instance_library(bandit_lp, include_original=True):
    """
    Return a dictionary containing the original rmab_instances library plus the
    extra instances in this file.
    """
    library = {}
    if include_original:
        from rmab_instances import build_instance_library

        library.update(build_instance_library(bandit_lp))
    library.update(build_extra_known_model_instance_library(bandit_lp))
    return library


def extra_instance_metadata_dataframe(instance_library):
    rows = []
    for spec in instance_library.values():
        rows.append(
            {
                "instance": spec.name,
                "group": getattr(spec, "group", "original"),
                "S": spec.bandit.S,
                "A": spec.bandit.A,
                "default_alpha": spec.default_alpha,
                "source": spec.source,
                "notes": spec.notes,
                "transition_summary": getattr(spec, "transition_summary", ""),
                "reward_summary": getattr(spec, "reward_summary", ""),
            }
        )
    return rows


def print_instance_overview(instance_library):
    """
    Print a compact human-readable overview of S, A, P structure and rewards.
    """
    for spec in instance_library.values():
        print(f"\n{spec.name}")
        print(f"  group: {getattr(spec, 'group', 'original')}")
        print(f"  S={spec.bandit.S}, A={spec.bandit.A}, alpha={spec.default_alpha}")
        print(f"  P: {getattr(spec, 'transition_summary', spec.notes)}")
        print(f"  R: {getattr(spec, 'reward_summary', '')}")


if __name__ == "__main__":
    from pathlib import Path
    import sys
    import pandas as pd

    CODE_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(CODE_DIR))

    import bandit_lp

    library = build_extra_known_model_instance_library(bandit_lp)
    metadata_df = pd.DataFrame(extra_instance_metadata_dataframe(library))
    print(metadata_df[["instance", "group", "S", "A", "default_alpha"]])
    print(f"\nTotal extra known-model instances: {len(library)}")
