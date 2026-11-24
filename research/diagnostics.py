"""Secondary descriptive analyses using frozen predictions; never writes policy."""

import numpy as np
import pandas as pd
from .summarize import RESULTS, P, point_stats, ci_mean
from .world import simulate
from src.research_estimate import estimate


def main():
    df = pd.read_csv(RESULTS / "final.csv").query("eligible").copy()
    ref = df.huber - df.truth
    rows = []
    for density in [5200, 6500, 7700, 8500, 9200]:
        error = df.huber * density / 7700 - df.truth
        delta = error.abs() - ref.abs()
        low, high = ci_mean(delta)
        rows.append(
            dict(
                assumed_kcal_kg=density,
                n=len(df),
                **point_stats(error),
                paired_mae_change=delta.mean(),
                low=low,
                high=high,
            )
        )
    pd.DataFrame(rows).to_csv(RESULTS / "assumed_density.csv", index=False)
    rows = []
    for split in ["development", "validation", "final", "shift"]:
        n = pd.read_csv(RESULTS / f"{split}.csv").shape[0]
        diffs = []
        for i in range(n):
            x, y, *_ = simulate(split, i)
            diffs.extend(np.diff(y)[np.diff(x) == 1])
        rows.append(
            dict(
                split=split,
                adjacent_pairs=len(diffs),
                median_abs_change=np.median(np.abs(diffs)),
                p90_abs_change=np.quantile(np.abs(diffs), 0.9),
                sd=np.std(diffs),
            )
        )
    pd.DataFrame(rows).to_csv(RESULTS / "scale_variation.csv", index=False)
    d = np.arange(42)
    readings = 76 - 0.04 * d
    output = estimate(readings)
    rows = [
        dict(world="A", tissue_slope=-0.04, water_slope=0, density=7700, balance=-308),
        dict(
            world="B", tissue_slope=-0.02, water_slope=-0.02, density=5500, balance=-110
        ),
    ]
    for row in rows:
        np.testing.assert_allclose(
            readings, 76 + (row["tissue_slope"] + row["water_slope"]) * d
        )
        row.update(
            reported=output["kcal_per_day"], kg_first=readings[0], kg_last=readings[-1]
        )
    pd.DataFrame(rows).to_csv(RESULTS / "identifiability.csv", index=False)


if __name__ == "__main__":
    main()
