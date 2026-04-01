"""
rendering.py – GUI rendering helpers for SponsorshipCaseManagerEnv
Provides both a pygame live-view and a matplotlib static screenshot.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # uses non-GUI backend (no pop-up window)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def render_static_frame(students, week, max_weeks, save_path="static_frame.png"):
    """
    Generate a static PNG showing the environment state – used for the
    'random action' demo screenshot required by the assignment rubric.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7),
                             facecolor="#0F1423",
                             gridspec_kw={"width_ratios": [3, 1]})
    fig.suptitle(
        f"ANLM Sponsorship Case Manager  |  Week {week}/{max_weeks}\n"
        "Random-Action Demo (no model – pure exploration)",
        color="#DCB84A", fontsize=13, fontweight="bold"
    )

    ax_table = axes[0]
    ax_stats = axes[1]
    ax_table.set_facecolor("#0F1423")
    ax_stats.set_facecolor("#0F1423")
    ax_table.axis("off")
    ax_stats.axis("off")

    # ---- Build table data ----
    col_labels = ["ID", "Letter", "Payment", "Photo", "Report", "Risk", "Donor\nScore"]
    cell_data  = []
    cell_colors = []

    G = "#4EC97A"; R = "#DC5050"; Y = "#F0C83C"; W = "#C8C8D8"; B = "#283050"

    for s in students:
        row = [
            f"{s['id']:02d}",
            "✓" if s["letter_done"]  else "✗",
            "✓" if s["payment_done"] else "✗",
            "✓" if s["photo_done"]   else "✗",
            "✓" if s["report_done"]  else "✗",
            "⚠" if s["at_risk"]      else "–",
            f"{s['donor_score']:.2f}",
        ]
        done_color  = lambda d: G if d else R
        score_c = G if s["donor_score"] > 0.7 else (Y if s["donor_score"] > 0.4 else R)
        cell_data.append(row)
        cell_colors.append([
            B, done_color(s["letter_done"]), done_color(s["payment_done"]),
            done_color(s["photo_done"]), done_color(s["report_done"]),
            Y if s["at_risk"] else B, score_c
        ])

    tbl = ax_table.table(
        cellText=cell_data,
        colLabels=col_labels,
        cellColours=cell_colors,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#3A4A70")
        cell.set_text_props(color="white" if r > 0 else "#DCB84A",
                            fontweight="bold" if r == 0 else "normal")

    # ---- Stats panel ----
    compliant  = sum(1 for s in students
                     if s["letter_done"] and s["payment_done"]
                     and s["photo_done"] and s["report_done"])
    at_risk    = sum(1 for s in students if s["at_risk"])
    avg_score  = np.mean([s["donor_score"] for s in students])

    metrics = [
        ("Fully Compliant",  f"{compliant}/{len(students)}", G),
        ("At Risk",          str(at_risk),                   Y if at_risk > 0 else G),
        ("Avg Donor Score",  f"{avg_score:.3f}",
         G if avg_score > 0.7 else (Y if avg_score > 0.4 else R)),
        ("Week Progress",    f"{week}/{max_weeks}",           W),
    ]

    for i, (label, val, color) in enumerate(metrics):
        y_pos = 0.85 - i * 0.22
        ax_stats.text(0.5, y_pos + 0.07, label,
                      transform=ax_stats.transAxes,
                      ha="center", va="center",
                      color="#9BAAC8", fontsize=9)
        ax_stats.text(0.5, y_pos - 0.02, val,
                      transform=ax_stats.transAxes,
                      ha="center", va="center",
                      color=color, fontsize=20, fontweight="bold")

    legend_patches = [
        mpatches.Patch(color=G, label="Done / On-track"),
        mpatches.Patch(color=R, label="Missed / Low score"),
        mpatches.Patch(color=Y, label="At risk / Warning"),
    ]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=3, facecolor="#1A2540", edgecolor="#3A4A70",
               labelcolor="white", fontsize=9, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.06, 1, 0.93])
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[render] Static frame saved → {save_path}")
    return save_path


def plot_training_curves(results_dict, save_path="training_curves.png"):
    """
    Plot cumulative reward curves for all four algorithms side-by-side.
    results_dict = { "DQN": [ep_rewards], "REINFORCE": [...], "A2C": [...], "PPO": [...] }
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor="#0F1423")
    fig.suptitle("Cumulative Reward per Episode – All Algorithms",
                 color="#DCB84A", fontsize=14, fontweight="bold")

    colors = {"DQN": "#4EC97A", "REINFORCE": "#5B9EF5",
              "PPO": "#C96BCC"}

    for ax, (name, rewards) in zip(axes.flatten(), results_dict.items()):
        ax.set_facecolor("#101828")
        episodes = np.arange(1, len(rewards) + 1)
        # Smooth with rolling mean
        window = max(1, len(rewards) // 20)
        smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")
        ax.plot(episodes, rewards, alpha=0.25, color=colors[name], linewidth=0.8)
        ax.plot(np.arange(window, len(rewards)+1), smoothed,
                color=colors[name], linewidth=2.2, label=f"{name} (smoothed)")
        ax.set_title(name, color=colors[name], fontweight="bold")
        ax.set_xlabel("Episode", color="#9BAAC8", fontsize=8)
        ax.set_ylabel("Cumulative Reward", color="#9BAAC8", fontsize=8)
        ax.tick_params(colors="#9BAAC8", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#3A4A70")
        ax.legend(fontsize=8, facecolor="#1A2540", edgecolor="#3A4A70",
                  labelcolor="white")
        ax.grid(True, alpha=0.15, color="#3A4A70")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[render] Training curves saved → {save_path}")
