"""Training-only global popularity rankings and exact-five candidate restriction.

No validation labels are accepted by fit_popularity. Each score variant uses the
same deterministic tie-break: distinct training users, training airing count,
then numeric program ID, descending for the first two and ascending for the ID.
"""

from collections import Counter
from dataclasses import dataclass
import time

from validation import EvaluationProtocol, training_input

VARIANTS = ("distinct_users", "qualifying_airings", "watched_seconds")


@dataclass(frozen=True)
class PopularityModel:
    candidates: tuple
    distinct_users: dict
    qualifying_airings: dict
    watched_seconds: dict
    training_sha256: str
    training_window_utc: tuple
    training_rows: int
    training_runtime_seconds: float

    def ranking(self, variant):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown popularity variant: {variant}")
        scores = {"distinct_users": self.distinct_users,
                  "qualifying_airings": self.qualifying_airings,
                  "watched_seconds": self.watched_seconds}[variant]
        return tuple(sorted(self.candidates,
                            key=lambda item: (-scores.get(item, 0),
                                              -self.distinct_users.get(item, 0),
                                              -self.qualifying_airings.get(item, 0), int(item))))


def fit_popularity(config, candidates=None):
    """Compute three all-user statistics using ONLY registered training rows."""
    started = time.monotonic()
    source = training_input(config)
    candidates = tuple(candidates) if candidates is not None else EvaluationProtocol().candidates
    users_by_program = {}
    airs, seconds = Counter(), Counter()
    rows = 0
    for row in source.rows():
        rows += 1
        users_by_program.setdefault(row.program_id, set()).add(row.user_id)
        airs[row.program_id] += row.qualifying_airings
        seconds[row.program_id] += row.watched_seconds
    distinct = {item: len(user_set) for item, user_set in users_by_program.items()}
    return PopularityModel(tuple(sorted(set(candidates), key=int)), distinct, dict(airs), dict(seconds),
                           source.sha256, tuple(source.window_utc), rows,
                           time.monotonic() - started)


def predictions(model, variant, cohort):
    ranking = model.ranking(variant)
    if len(model.candidates) < 5:
        raise ValueError("Candidate pool must contain at least five programs")
    return {u: ranking[:5] for u in cohort}


if __name__ == "__main__":
    protocol = EvaluationProtocol()
    model = fit_popularity(protocol.config, protocol.candidates)
    print({"training_rows": model.training_rows, "training_seconds": model.training_runtime_seconds,
           "variants": {name: model.ranking(name)[:5] for name in VARIANTS}})
