# 補助文書03 のスクリプトで共通に使う設定
# すべて架空のデータで動く。図は ../fig/ に保存する。
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "fig")
os.makedirs(FIG, exist_ok=True)

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "IPAGothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

NAVY = "#1B4965"
CYAN = "#3AA6B9"
AMBER = "#E8A33D"
RED = "#C8553D"
GREY = "#8A8F98"


def savefig(fig, name):
    path = os.path.join(FIG, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path
