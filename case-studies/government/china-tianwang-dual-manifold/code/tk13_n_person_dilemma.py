"""
敵の多様体 補助文書03 コード13：N人囚人のジレンマと、メカニズムデザイン
2,850人の幹部。正直に報告すると、全体の利益 B（天網の精度の向上）が人数に比例して増えるが、本人は損失 c（昇進の停止など）を負う。
各人の利得 = B × (正直な人の割合) − c × (自分が正直なら1)。
制度（正直な報告への報奨 r、または損失の免除）を加えると、ナッシュ均衡が変わることを確かめる。数値は架空のもの。
"""
N, B, c = 2850, 10.0, 1.0
def payoff(honest_share, me_honest, reward=0.0):
    return B * honest_share - c * me_honest + reward * me_honest
for reward in [0.0, 1.5]:
    # 全員が隠しているとき、自分だけ正直にすると得か？（逸脱の誘因）
    gain_dev = payoff(1 / N, 1, reward) - payoff(0, 0, reward)
    # 全員が正直なとき、自分だけ隠すと得か？
    gain_hide = payoff((N - 1) / N, 0, reward) - payoff(1.0, 1, reward)
    eq = "全員が隠す" if gain_dev < 0 else ("全員が正直" if gain_hide < 0 else "混合")
    print(f"報奨 r={reward}: 隠す中で自分だけ正直にする利得の変化 {gain_dev:+.4f}、正直な中で自分だけ隠す利得の変化 {gain_hide:+.4f} → 均衡：{eq}")
print(f"社会的最適（全員が正直）の一人あたり利得 {payoff(1.0, 1):.2f}、全員が隠すときの利得 {payoff(0, 0):.2f}")
print("→ 報奨が個人の損失を上回れば（誘因両立性）、正直に報告することが各自の最適な選択になる。")
