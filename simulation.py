"""Պոկերի river-փուլի պարզեցված սիմուլյացիա՝ բլեֆի տվյալներ ստանալու համար."""
import numpy as np
import pandas as pd


def generate_hands(n=20000, seed=42):
    rng = np.random.default_rng(seed)
    strength = rng.uniform(0, 1, n)          # ձեռքի ուժը [0, 1]
    wet_board = rng.integers(0, 2, n)        # 1՝ վտանգավոր սեղան (draw-եր կան)
    in_position = rng.integers(0, 2, n)      # 1՝ խաղացողը դիրքում է

    rows = []
    for i in range(n):
        s = strength[i]
        if s > 0.70:                         # ուժեղ ձեռք՝ value bet
            bet, bluff = True, 0
            size = rng.choice([0.5, 0.75, 1.0], p=[0.2, 0.4, 0.4])
        elif s < 0.25 and rng.random() < 0.5:  # թույլ ձեռք՝ բլեֆ
            bet, bluff = True, 1
            size = rng.choice([0.5, 0.75, 1.0], p=[0.4, 0.4, 0.2])
        else:
            continue                         # check, այս դեպքը չենք դիտարկում
        rows.append((s, wet_board[i], in_position[i], size, bluff))

    return pd.DataFrame(rows, columns=[
        "strength", "wet_board", "in_position", "bet_ratio", "is_bluff"])
