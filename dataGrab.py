from bs4 import BeautifulSoup
from requests import get
import numpy as np
import pandas as pd
from functools import lru_cache


# gets list of each team + their conference/division for a given nfl season
@lru_cache(maxsize=None)
def get_teams(year):
    url = 'https://www.pro-football-reference.com/years/' + str(year) + '/'
    html = BeautifulSoup(get(url).text, features='lxml')
    afc_teams = _parse_standings(html.select('table[id=AFC] > tbody > tr'), 'AFC')
    nfc_teams = _parse_standings(html.select('table[id=NFC] > tbody > tr'), 'NFC')
    return pd.DataFrame(afc_teams + nfc_teams)


# determines division of each team in a conference standings table
def _parse_standings(rows, conference):
    teams = []
    division = ''
    for row in rows:
        if 'class' in row.attrs:
            division = row.text.strip()
        else:
            teams.append({'Team': row.select('a')[0].text, 'Conference': conference, 'Division': division})
    return teams


# gets schedule and game log
@lru_cache(maxsize=None)
def get_schedule(year):
    url = 'https://www.pro-football-reference.com/years/' + str(year) + '/games.htm'
    df = pd.read_html(str(BeautifulSoup(get(url).text, features='lxml').select('table[id=games]')))[0]
    df.columns = ['Week', 'Day', 'Date', 'Time', 'Winner', 'At', 'Loser', 'Box', 'PtsW', 'PtsL', 'Del', 'Del', 'Del',
                  'Del']
    df = df[df['Date'] != 'Playoffs']
    df = df[df['Week'].apply(lambda x: x.isnumeric())]
    df = df[[x for x in list(df) if x not in ['Box', 'Del', 'Day']]]
    is_home = (df['At'].values != '@')
    df['Home'] = np.where(is_home, df['Winner'].values, df['Loser'].values)
    df['Away'] = np.where(is_home, df['Loser'].values, df['Winner'].values)
    df['HomePts'] = np.nan_to_num(np.where(is_home, df['PtsW'].values, df['PtsL'].values).astype(float)).astype(int)
    df['AwayPts'] = np.nan_to_num(np.where(is_home, df['PtsL'].values, df['PtsW'].values).astype(float)).astype(int)
    df['Played'] = df['PtsW'].notnull()
    df['Week'] = df['Week'].astype(int)
    return df[['Week', 'Home', 'Away', 'HomePts', 'AwayPts', 'Played']].reset_index(drop=True)