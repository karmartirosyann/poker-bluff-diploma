"""Բայեսյան բազային մոդել՝ հիմնված pot odds-ի վրա."""
import numpy as np


def gto_bluff_share(bet_ratio):
    """Թեորիական բլեֆի մասնաբաժին՝ alpha = B / (P + 2B), որտեղ bet_ratio = B/P."""
    return bet_ratio / (1 + 2 * bet_ratio)


def bayes_posterior(bet_ratio, p_bluff_prior, lik_bluff, lik_value):
    """P(բլեֆ | խաղադրույք) Բայեսի բանաձևով."""
    num = lik_bluff * p_bluff_prior
    den = num + lik_value * (1 - p_bluff_prior)
    return num / den
