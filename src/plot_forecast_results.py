import matplotlib.pyplot as plt

horizons = [1, 2, 3, 4, 5]

f1_scores = [
    0.9793,
    0.9710,
    0.9624,
    0.9532,
    0.9440
]

roc_auc = [
    0.9937,
    0.9859,
    0.9759,
    0.9661,
    0.9563
]

plt.figure(figsize=(8, 5))

plt.plot(
    horizons,
    f1_scores,
    marker="o",
    label="F1 Score"
)

plt.plot(
    horizons,
    roc_auc,
    marker="o",
    label="ROC-AUC"
)

plt.xlabel("Forecast Horizon")
plt.ylabel("Score")

plt.title(
    "Multi-Step Attack Forecasting Performance"
)

plt.xticks(horizons)

plt.ylim(0.90, 1.01)

plt.grid(True, alpha=0.3)

plt.legend()

plt.tight_layout()

plt.savefig(
    "data/attack_forecast_performance.png",
    dpi=200
)

plt.show()

print(
    "Saved: data/attack_forecast_performance.png"
)