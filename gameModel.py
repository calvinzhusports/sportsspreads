import pandas as pd
import hardcodedData as hd
from scipy.special import expit


ratings = pd.read_csv('inpredict_NFL_2020.csv')
ratings['team_name'] = ratings['team'].apply(lambda x: hd.inpredictable_mappings[x])
rankings = ratings[ratings['DT'] == ratings['DT'].max()].set_index('team_name')['gpf']


def compute_game_prob(home_team, away_team, home_field=True):
    home_ranking = rankings[home_team]
    away_ranking = rankings[away_team]
    if home_field:
        diff = home_ranking - away_ranking + 1.4
    else:
        diff = home_ranking - away_ranking
    prob_model = expit(0.145 * diff)
    return prob_model
