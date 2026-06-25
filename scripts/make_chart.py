import os
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# df = pd.read_csv("us_full.csv")

# order = ["52_weeks_high", "price_jump_up", "gap_up", "volume_spike_up",
#          "52_weeks_low", "price_jump_down", "gap_down", "volume_spike_down"]
# df["signal"] = pd.Categorical(df["signal"], categories=order, ordered=True)
# df = df.sort_values("signal")

# os.makedirs("docs/img", exist_ok=True)

# ax = df.plot.bar(x="signal", y=["exc_5", "exc_20", "exc_60"], figsize=(9, 4))
# ax.axhline(0, color="black", lw=0.8)
# ax.set_ylabel("시장 대비 초과수익 (%)")
# ax.set_title("신호별 향후 초과수익 (US, 1년)")
# plt.xticks(rotation=30, ha="right")
# plt.tight_layout()
# plt.savefig("docs/img/exc_by_signal_US.png", dpi=120)
# print("saved")

boot20 = {"52주 신고가":     (1.72, 0.38, 3.04),
    "price_jump_up":  (0.26, -0.78, 1.30),
    "gap_up":         (0.56, -0.37, 1.51),
    "volume_spike_up":(0.39, -0.42, 1.26),
    "52주 신저가":     (0.67, -1.03, 2.47),
    "price_jump_down":(-0.08, -0.87, 0.79),
    "gap_down":       (-0.05, -0.89, 0.81),
    "volume_spike_down":(-0.02, -0.67, 0.67),
}

boot60 = {
    "52주 신고가":      (1.15, -1.37, 3.83),
    "price_jump_up":   (-0.55, -2.73, 1.70),
    "gap_up":          (-0.65, -2.41, 1.29),
    "volume_spike_up": (-0.79, -2.53, 1.06),
    "52주 신저가":      (-3.17, -5.97, -0.35),
    "price_jump_down": (-1.45, -3.18, 0.36),
    "gap_down":        (-1.08, -2.70, 0.73),
    "volume_spike_down":(-0.32, -1.90, 1.36),
}

labels = list(boot20)
pts = [boot60[k][0] for k in labels]
los = [boot60[k][1] for k in labels]
his = [boot60[k][2] for k in labels]
y = range(len(labels))
colors = ["crimson" if lo > 0 or hi < 0 else "gray" for lo, hi in zip(los, his)]

fig, ax = plt.subplots(figsize=(8, 5))
for i, (p, lo, hi, c) in enumerate(zip(pts, los, his, colors)):
    ax.plot([lo, hi], [i, i], color=c, lw=2)
    ax.plot(p, i, "o", color=c, ms=7)
ax.axvline(0, color="black", lw=0.8, ls="--")
ax.set_yticks(list(y))
ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.set_title("신호별 초과수익 신뢰구간 (95%, 60일)")
plt.tight_layout()
os.makedirs("docs/img", exist_ok=True)
plt.savefig("docs/img/bootstrap_ci_60d.png", dpi=120)
print("saved")