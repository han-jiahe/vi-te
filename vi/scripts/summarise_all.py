import glob
import os
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 220)

frames = []
for f in sorted(glob.glob("output/vi/*/task1_ce_summary.csv")):
    d = pd.read_csv(f)
    d["match_id"] = os.path.basename(os.path.dirname(f))
    frames.append(d)
df = pd.concat(frames, ignore_index=True)

print("比赛数", df.match_id.nunique(), " 行数", len(df))
print()
print("=== Task 1: 每场每半场 Ψ_global (nats) ===")
p = df.pivot_table(index="match_id", columns="half", values="psi_global", aggfunc="mean")
print(p.round(3).to_string())
print()
print("Ψ_global  全距 %.3f ~ %.3f   均值 %.3f"
      % (df.psi_global.min(), df.psi_global.max(), df.psi_global.mean()))
print("Ψ_rolling 全距 %.3f ~ %.3f   均值 %.3f"
      % (df.psi_rolling_mean.min(), df.psi_rolling_mean.max(),
         df.psi_rolling_mean.mean()))
print()
print("=== 有效秒（锚点 4405）===")
print(df.groupby("match_id")[["n_seconds", "n_windows"]].sum().to_string())
print()
print("=== 控球率 ===")
print(df.pivot_table(index="match_id", columns="side",
                     values="possession_rate").round(3).to_string())
