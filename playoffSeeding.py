import numpy as np
import pandas as pd
from dataGrab import get_teams


def get_playoff_seeding(game_log, year):
    home_list = [x + [True] for x in game_log[['Home', 'Away', 'HomePts', 'AwayPts']].values.tolist()]
    away_list = [x + [False] for x in game_log[['Away', 'Home', 'AwayPts', 'HomePts']].values.tolist()]
    full_game_log = pd.DataFrame(home_list + away_list, columns=['Team', 'Opponent', 'Points', 'OppPoints', 'IsHome'])
    full_game_log['Wins'] = (full_game_log['Points'] > full_game_log['OppPoints']).astype(int)
    full_game_log['Losses'] = (full_game_log['Points'] < full_game_log['OppPoints']).astype(int)
    full_game_log['Ties'] = (full_game_log['Points'] == full_game_log['OppPoints']).astype(int)
    total_team_wins = full_game_log.groupby(['Team']).agg({'Wins': 'sum', 'Losses': 'sum', 'Ties': 'sum'}).reset_index()
    total_team_wins = total_team_wins.rename(columns={'Wins': 'Total Wins', 'Losses': 'Total Losses',
                                                      'Ties': 'Total Ties'})
    opp_wins = total_team_wins.rename(columns={'Team': 'Opponent', 'Total Wins': 'Opponent Total Wins',
                                               'Total Losses': 'Opponent Total Losses',
                                               'Total Ties': 'Opponent Total Ties'})
    full_game_log = pd.merge(pd.merge(full_game_log, total_team_wins, on='Team'), opp_wins, on='Opponent')
    team_info = get_teams(year).set_index('Team')
    full_game_log['Division'] = full_game_log['Team'].apply(lambda x: team_info['Division'][x])
    full_game_log['Conference'] = full_game_log['Team'].apply(lambda x: team_info['Conference'][x])
    full_game_log['OppDivision'] = full_game_log['Opponent'].apply(lambda x: team_info['Division'][x])
    full_game_log['OppConference'] = full_game_log['Opponent'].apply(lambda x: team_info['Conference'][x])
    team_wins = full_game_log.groupby(['Team', 'Division'])['Wins'].sum() + 0.5 * full_game_log.groupby(
        ['Team', 'Division'])['Ties'].sum()
    top_teams = team_wins[team_wins.groupby('Division').rank(method='min', ascending=False).values == 1]
    div_winners = set()
    return_bad_tie = False
    for d, g in top_teams.groupby('Division'):
        div_tie, bad_tie = breakDivisionalTie(full_game_log,
                                              [{'Team': t, 'Division': d}
                                               for t in g.index.get_level_values('Team').values])
        div_winners.add(div_tie)
        return_bad_tie = bad_tie or return_bad_tie
    remaining_teams = set(full_game_log['Team'].values) - div_winners
    if year >= 2020:
        seed_dict, bad_tie = getSeeds(full_game_log, ([1, 2, 3, 4], [5, 6, 7]), (div_winners, remaining_teams))
    else:
        seed_dict, bad_tie = getSeeds(full_game_log, ([1, 2, 3, 4], [5, 6]), (div_winners, remaining_teams))
    return_bad_tie = bad_tie or return_bad_tie
    seed_df = pd.DataFrame(seed_dict)
    return seed_df.set_index(['Conference', 'Seed'])['Team'], return_bad_tie


def resolveWinPercentage(gamelog):
    grouped = gamelog.groupby(['Team']).agg({'Wins': 'sum', 'Losses': 'sum', 'Ties': 'sum'}).reset_index()
    return resolveMaxWins(grouped, 'Wins', 'Losses', 'Ties')


def resolveScheduleStrength(gamelog):
    grouped = gamelog.groupby(['Team']).agg({'Opponent Total Wins': 'sum',
                                             'Opponent Total Losses': 'sum',
                                             'Opponent Total Ties': 'sum'}).reset_index()
    return resolveMaxWins(grouped, 'Opponent Total Wins', 'Opponent Total Losses', 'Opponent Total Ties')


def resolveMaxWins(df, wincol, losscol, tiecol):
    percent = ((df[wincol].values + (0.5 * df[tiecol]).values) / (
                df[wincol].values + df[losscol].values + df[tiecol].values))
    ismax = (percent == percent.max())
    return df[ismax]['Team'].values


def resolveH2HSweep(gamelog):
    teams = set(gamelog['Team'].values)
    if len(teams) == 2:
        return resolveWinPercentage(gamelog)
    grouped = gamelog.groupby(['Team']).agg({'Wins': 'sum', 'Losses': 'sum'}).reset_index()
    undefeated = grouped['Wins'].values == (len(teams) - 1)
    if undefeated.sum() > 0:
        return grouped[undefeated]['Team'].values
    swept = grouped['Losses'].values == (len(teams) - 1)
    if swept.sum() > 0:
        return grouped[~swept]['Team'].values
    return list(teams)


def getCommonOpponents(gamelog, teams):
    teamgames = gamelog[gamelog['Team'].isin(teams)].groupby('Team')
    return set.intersection(*[set(g['Opponent'].values) for t, g in teamgames])


# define filters
def teamfilter(df, lst):
    return df['Team'].isin(lst)


def h2hfilter(df, lst):
    return teamfilter(df, lst) & df['Opponent'].isin(lst)


def sweepfilter(df, lst):
    x = h2hfilter(df, lst)
    y = len(set(df[x]['Team'].values).union(set(df[x]['Opponent'].values))) == len(lst)
    return x & y


def divisionfilter(df, lst):
    return teamfilter(df, lst) & (df['Division'] == df['OppDivision'])


def cgfilter(df, lst, n):
    common = getCommonOpponents(df, lst)
    return teamfilter(df, lst) & (df['Opponent'].isin(common)) & (len(common) >= n)


def cgfilter_min1(df, lst):
    return cgfilter(df, lst, 1)


def cgfilter_min4(df, lst):
    return cgfilter(df, lst, 4)


def conferencefilter(df, lst):
    return teamfilter(df, lst) & (df['Conference'] == df['OppConference'])


def victoryfilter(df, lst):
    return teamfilter(df, lst) & (df['Wins'] == 1)


# define tiebreaker steps
divsteps = [{'Filter': h2hfilter, 'Resolution': resolveWinPercentage},
            {'Filter': divisionfilter, 'Resolution': resolveWinPercentage},
            {'Filter': cgfilter_min1, 'Resolution': resolveWinPercentage},
            {'Filter': conferencefilter, 'Resolution': resolveWinPercentage},
            {'Filter': victoryfilter, 'Resolution': resolveScheduleStrength},
            {'Filter': teamfilter, 'Resolution': resolveScheduleStrength}]

wcsteps = [{'Filter': sweepfilter, 'Resolution': resolveH2HSweep},
           {'Filter': conferencefilter, 'Resolution': resolveWinPercentage},
           {'Filter': cgfilter_min4, 'Resolution': resolveWinPercentage},
           {'Filter': victoryfilter, 'Resolution': resolveScheduleStrength},
           {'Filter': teamfilter, 'Resolution': resolveScheduleStrength}]


def getSeeds(gamelog, seed_groups, winner_groups):
    wins = gamelog.groupby(['Team', 'Conference', 'Division'])['Wins'].sum()
    seeding = []
    return_bad_tie = False
    for conf, g in wins.groupby('Conference'):
        for sg, seed_group in enumerate(seed_groups):
            filtered = g[[x in winner_groups[sg] for x in g.index.get_level_values('Team').values]]
            c_rank = filtered.rank(method='min', ascending=False).values
            for s, seed in enumerate(seed_group):
                indices = [x for x in filtered[c_rank <= s + 1].index.values if
                           x[0] not in [x['Team'] for x in seeding]]
                winner, bad_tie = breakWildCardTie(gamelog, [{'Team': t, 'Division': d} for t, c, d in indices])
                return_bad_tie = bad_tie or return_bad_tie
                seeding.append({'Conference': conf, 'Team': winner, 'Seed': seed})
    return seeding, return_bad_tie


# breaks ties using divisional rules
def breakDivisionalTie(gamelog, tiedteams):
    if len(tiedteams) == 1:
        return tiedteams[0]['Team'], False
    return breakTies(gamelog, divsteps, tiedteams, breakDivisionalTie)


# breaks ties using wildcard rules
def breakWildCardTie(gamelog, tiedteams):
    divisions = set([team['Division'] for team in tiedteams])
    if len(divisions) == 1:
        return breakDivisionalTie(gamelog, tiedteams)
    currentteams = set()
    return_bad_tie = False
    for division in divisions:
        div_tie, bad_tie = breakDivisionalTie(gamelog, [x for x in tiedteams if x['Division'] == division])
        currentteams.add(div_tie)
        return_bad_tie = return_bad_tie or bad_tie
    bt, bad_tie = breakTies(gamelog, wcsteps, [x for x in tiedteams if x['Team'] in currentteams], breakWildCardTie)
    return bt, (bad_tie or return_bad_tie)


# breaks ties between 2 or more teams by applying an ordered list of filters/functions to a game log
def breakTies(gamelog, steps, tiedteams, caller):
    remainder = [x['Team'] for x in tiedteams]
    for step in steps:
        filtered = gamelog[step['Filter'](gamelog, remainder)]
        if not filtered.empty:
            remainder = step['Resolution'](filtered)
        if len(remainder) == 1:
            return remainder[0], False
        elif len(remainder) == 2 and len(tiedteams) != 2:
            return caller(gamelog, [x for x in tiedteams if x['Team'] in remainder])
    return np.random.choice(remainder), True
