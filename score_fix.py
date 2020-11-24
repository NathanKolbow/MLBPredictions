import numpy as np
import pandas as pd

df = pd.read_csv("project_data.csv")
df = df.sort_values(by=['game_date', 'home_team', 'at_bat_number', 'pitch_number'])

curr_date = df['game_date'].iloc[0]
curr_home = df['home_team'].iloc[0]
curr_at_bat = df['at_bat_number'].iloc[0]
for i in range(1, df.shape[0]):
    print(f'{i}/{df.shape[0]}', end='\r')
    if curr_date == df['game_date'].iloc[i] and \
       curr_home == df['home_team'].iloc[i] and \
       curr_at_bat == df['at_bat_number'].iloc[i]:
        df['post_bat_score'].iloc[i-1] = df['bat_score'].iloc[i]
    else:
        if df['at_bat_number'].iloc[i] == curr_at_bat + 1:
            df['post_bat_score'].iloc[i-1] = df['bat_score'].iloc[i]
            
        curr_date = df['game_date'].iloc[i]
        curr_home = df['home_team'].iloc[i]
        curr_at_bat = df['at_bat_number'].iloc[i]
        
df.to_csv("project_data_corrected.csv")