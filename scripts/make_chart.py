import os
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv("us_full.csv")

order = ["52_weeks_high", "price_jump_up", "gap_up", "volume_spike_up",
         "52_weeks_low", "price_jump_down", "gap_down", "volume_spike_down"]
df["signal"] = pd.Categorical(df["signal"], categories=order, ordered=True)
df = df.sort_values("signal")

os.makedirs("docs/img", exist_ok=True)

ax = df.plot.bar(x="signal", y=["exc_5", "exc_20", "exc_60"], figsize=(9, 4))
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("시장 대비 초과수익 (%)")
ax.set_title("신호별 향후 초과수익 (US, 1년)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("docs/img/exc_by_signal_US.png", dpi=120)
print("saved")
