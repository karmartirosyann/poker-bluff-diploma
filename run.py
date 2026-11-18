import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from simulation import generate_hands
from baseline import gto_bluff_share

df = generate_hands()
print("Ձեռքերի քանակ:", len(df))
print("Բլեֆի մասնաբաժին:", round(df.is_bluff.mean(), 3))
print(df.groupby("bet_ratio").is_bluff.mean().round(3))

X = df[["bet_ratio", "wet_board", "in_position"]]
y = df["is_bluff"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

p_base = gto_bluff_share(Xte["bet_ratio"].values)
print("\nԲազային (GTO) AUC:", round(roc_auc_score(yte, p_base), 3),
      "logloss:", round(log_loss(yte, p_base), 3))

lr = LogisticRegression().fit(Xtr, ytr)
p_lr = lr.predict_proba(Xte)[:, 1]
print("Լոգիստիկ ռեգրեսիա AUC:", round(roc_auc_score(yte, p_lr), 3),
      "logloss:", round(log_loss(yte, p_lr), 3),
      "accuracy:", round(accuracy_score(yte, p_lr > 0.5), 3))
print("Գործակիցներ:", dict(zip(X.columns, lr.coef_[0].round(3))))
