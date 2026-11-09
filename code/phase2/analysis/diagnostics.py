"""Aggregate paired validation diagnostics without including document text."""
from collections import Counter
from itertools import combinations

import numpy as np

from ..ensemble import ensemble_documents
from ..metrics import strict_metrics


def false_positive_categories(gold, predictions):
    counts = Counter()
    for doc_id, spans in gold.items():
        true = {tuple(span[:3]) for span in spans}
        for start, end, label in {tuple(span) for span in predictions[doc_id]} - true:
            overlap = [(a, b, category) for a, b, category in true if a < end and start < b]
            if any(a == start and b == end for a, b, _ in overlap):
                counts["Wrong entity label"] += 1
            elif any(category == label for _, _, category in overlap):
                counts["Incorrect boundary"] += 1
            elif overlap:
                counts["Boundary and label mismatch"] += 1
            else:
                counts["Spurious entity"] += 1
    return dict(counts)


def paired_bootstrap(gold, baseline, ensemble, seed):
    rng = np.random.default_rng(seed)
    keys = list(gold)
    differences = []
    for _ in range(500):
        draw = list(enumerate(rng.choice(keys, size=len(keys), replace=True)))
        # Distinct keys retain repeated documents in each resample.
        boot_gold, boot_base, boot_ensemble = (
            {i: records[key] for i, key in draw} for records in (gold, baseline, ensemble)
        )
        base = strict_metrics(boot_gold, boot_base)[0]["macro_f1"]
        vote = strict_metrics(boot_gold, boot_ensemble)[0]["macro_f1"]
        differences.append(vote - base)
    low, high = np.quantile(differences, (0.025, 0.975))
    return {
        "mean_difference": float(np.mean(differences)), "ci_low": float(low),
        "ci_high": float(high), "fraction_positive": float(np.mean(np.array(differences) > 0)),
    }


def summarize_ensemble(gold, members, strategy, seed):
    baseline = max(members, key=lambda name: strict_metrics(gold, members[name])[0]["macro_f1"])
    voted = ensemble_documents(members, strategy)
    pairwise = {}
    for first, second in combinations(members, 2):
        sets = [(set(map(tuple, members[first][key])), set(map(tuple, members[second][key])))
                for key in gold]
        shared = sum(len(a & b) for a, b in sets)
        union = sum(len(a | b) for a, b in sets)
        pairwise[f"{first}/{second}"] = {"shared": shared, "jaccard": shared / union if union else 0.0}
    return {
        "baseline": baseline, "strategy": strategy, "pairwise": pairwise,
        "baseline_errors": false_positive_categories(gold, members[baseline]),
        "ensemble_errors": false_positive_categories(gold, voted),
        "bootstrap": paired_bootstrap(gold, members[baseline], voted, seed),
    }
