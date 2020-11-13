from dataGrab import get_schedule, get_teams
from gameModel import compute_game_prob
from playoffSeeding import get_playoff_seeding
import numpy as np
import pandas as pd


def simulate(year, trials=10, override_schedule=False):
    schedule = get_schedule(year)
    if override_schedule:
        print("Make sure 'Played' column is overridden to True for desired games")
        schedule = pd.read_csv('schedule_output.csv')
    else:
        schedule.to_csv('schedule_output.csv')
    remaining_games = schedule[-schedule['Played']]
    simulated_games = _simulate_regular_season(remaining_games, trials)
    season_outcomes, bad_ties = _determine_playoffs(schedule, simulated_games, year)
    playoff_outcomes = _simulate_playoffs(season_outcomes)
    playoff_outcomes = playoff_outcomes.reindex(get_teams(year)['Team'].values).T.fillna(0)
    playoff_outcomes = playoff_outcomes.replace({0: 'No Playoffs',
                                                    1: 'Wild Card Round Exit',
                                                    2: 'Divisional Round Exit',
                                                    3: 'Championship Round Exit',
                                                    4: 'Super Bowl Loser',
                                                    5: 'Super Bowl Winner'})
    print("Random coin tiebreaker activated in {} out of {} trials".format(bad_ties.sum(), trials))
    probabilities = (playoff_outcomes.apply(pd.value_counts) / trials).T
    get_div_winners = season_outcomes.reset_index()
    division_winners = get_div_winners[get_div_winners['Seed'] <= 4].set_index(['Conference', 'Seed'])
    probabilities['Division Winner'] = (division_winners.T.apply(pd.value_counts).sum(axis=1) / trials)
    probabilities = probabilities.fillna(0.0)
    probabilities.to_csv('probabilities_output.csv')
    season_outcomes.T.to_csv('seedings_output.csv')
    return probabilities


def _simulate_regular_season(games, trials):
    game_prob = {}
    for i, row in games.iterrows():
        game_prob[i] = compute_game_prob(row['Home'], row['Away'])
    game_prob = pd.Series(game_prob)
    random_nums = pd.DataFrame(np.random.uniform(size=(len(games.index), trials)), index=games.index)
    game_results = (random_nums.subtract(game_prob, axis=0) < 0.0)
    return game_results


def _determine_playoffs(schedule, simulated_games, year):
    played_games = schedule.loc[~schedule.index.isin(simulated_games.index)]
    trial_playoffs = {}
    bad_ties = {}
    for trial in simulated_games.columns:
        simulated_trial = schedule.loc[simulated_games.index]
        simulated_trial['HomePts'] = np.where(simulated_games[trial], 1, 0)
        simulated_trial['AwayPts'] = np.where(simulated_games[trial], 0, 1)
        trial_games = played_games.append(simulated_trial)
        playoff_seeds, bad_tie = get_playoff_seeding(trial_games, year)
        trial_playoffs[trial] = playoff_seeds
        bad_ties[trial] = bad_tie
    return pd.DataFrame(trial_playoffs), pd.Series(bad_ties)


def _simulate_playoffs(season_outcomes):
    playoff_outcomes = {}
    for trial in season_outcomes.columns:
        bracket = season_outcomes[trial].unstack(level=0)
        playoff_round = 1
        results = {v: playoff_round for v in season_outcomes[trial].values}
        remaining_bracket = bracket.copy()
        championship = False
        while not championship:
            round_winners = _simulate_playoff_round(remaining_bracket)
            playoff_round = playoff_round + 1
            for rw in round_winners:
                results[rw] = playoff_round
            if len(round_winners) == 1:
                championship = True
            else:
                new_ordering = bracket[bracket.isin(round_winners)].unstack().dropna().sort_index()
                remaining_bracket = pd.DataFrame({bracket.columns[0]: new_ordering[bracket.columns[0]].values,
                                                  bracket.columns[1]: new_ordering[bracket.columns[1]].values})
        playoff_outcomes[trial] = results.copy()
    return pd.DataFrame(playoff_outcomes)


def _simulate_playoff_round(bracket):
    if len(bracket.index) == 1:
        # finals
        prob = compute_game_prob(bracket.iloc[0][bracket.columns[0]],
                                 bracket.iloc[0][bracket.columns[1]],
                                 home_field=False)
        return [bracket.iloc[0][bracket.columns[0]]] if np.random.uniform() < prob else [bracket.iloc[0][bracket.columns[1]]]
    else:
        winners = []
        for conf in bracket.columns:
            remaining_teams = bracket[conf]
            max_num_games = int(2 ** np.ceil(np.log2(len(remaining_teams))) / 2)
            for i in range(max_num_games):
                if (2*max_num_games - i) > len(remaining_teams):
                    winners.append(remaining_teams.iloc[i])
                else:
                    home_team = remaining_teams.iloc[i]
                    away_team = remaining_teams.iloc[2*max_num_games - 1 - i]
                    prob = compute_game_prob(home_team, away_team)
                    winners.append(home_team if np.random.uniform() < prob else away_team)
        return winners
