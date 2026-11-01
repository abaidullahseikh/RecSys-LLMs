"""Per-user frequency baselines from Weeks 11–20 and common popularity fallback."""

from collections import Counter, defaultdict
from dataclasses import dataclass
import time

from popularity import PopularityModel
from validation import EvaluationProtocol, training_input

VARIANTS = ("qualifying_airings", "watched_seconds")


@dataclass(frozen=True)
class PersonalFrequencyModel:
    candidates: tuple
    user_airings: dict
    user_seconds: dict
    training_sha256: str
    training_window_utc: tuple
    training_rows: int
    training_runtime_seconds: float

    def user_ranking(self, user, variant, fallback):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown personal-frequency variant: {variant}")
        per_user = self.user_airings if variant == "qualifying_airings" else self.user_seconds
        scores = per_user.get(user, {})
        eligible = frozenset(self.candidates)
        personal = sorted((item for item in scores if item in eligible),
                          key=lambda item: (-scores[item], int(item)))
        seen = set(personal)
        # Repeat-allowed: prior programs stay in personal ranking when eligible.
        # Popularity fallback completes remaining slots and also serves cold users.
        personal.extend(item for item in fallback if item in eligible and item not in seen)
        return tuple(personal)


def fit_personal_frequency(config, candidates=None):
    started = time.monotonic()
    source = training_input(config)
    candidates = tuple(candidates) if candidates is not None else EvaluationProtocol().candidates
    airings, seconds = defaultdict(Counter), defaultdict(Counter)
    rows = 0
    for row in source.rows():
        rows += 1
        airings[row.user_id][row.program_id] += row.qualifying_airings
        seconds[row.user_id][row.program_id] += row.watched_seconds
    return PersonalFrequencyModel(tuple(sorted(set(candidates), key=int)),
                                  {u: dict(v) for u, v in airings.items()},
                                  {u: dict(v) for u, v in seconds.items()},
                                  source.sha256, tuple(source.window_utc), rows,
                                  time.monotonic() - started)


def predict(model, variant, cohort, popularity_model, fallback_variant="distinct_users"):
    if model.candidates != popularity_model.candidates:
        raise ValueError("Personal and popularity candidate pools differ")
    if model.training_sha256 != popularity_model.training_sha256 or model.training_window_utc != popularity_model.training_window_utc:
        raise ValueError("Personal and fallback training provenance differs")
    fallback = popularity_model.ranking(fallback_variant)
    result = {}
    for user in cohort:
        result[user] = model.user_ranking(user, variant, fallback)[:5]
    return result


if __name__ == "__main__":
    protocol = EvaluationProtocol()
    from popularity import fit_popularity
    global_model = fit_popularity(protocol.config, protocol.candidates)
    model = fit_personal_frequency(protocol.config, protocol.candidates)
    print({"training_rows": model.training_rows, "training_seconds": model.training_runtime_seconds,
           "variants": {name: model.user_ranking(protocol.cohort[0], name, global_model.ranking("distinct_users"))[:5]
                        for name in VARIANTS}})
