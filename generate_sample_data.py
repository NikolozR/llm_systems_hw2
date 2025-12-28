import pandas as pd
import numpy as np

np.random.seed(42)

data = {
    'MatchID': range(1, 101),
    'Opponent': np.random.choice(['Man City', 'Liverpool', 'Chelsea', 'Spurs', 'Man Utd', 'Aston Villa', 'Brighton'], size=100),
    'Venue': np.random.choice(['Home', 'Away'], size=100),
    'Possession': np.random.uniform(30, 75, size=100),
    'ShotsOnTarget': np.random.randint(0, 15, size=100),
    'Corners': np.random.randint(0, 12, size=100),
    'YellowCards': np.random.randint(0, 6, size=100),
    'InjuredStarters': np.random.randint(0, 5, size=100),
    'DaysRest': np.random.choice([3, 4, 6, 7], size=100),
    'Weather': np.random.choice(['Rain', 'Clear', 'Windy', np.nan], size=100),
    'ArsenalWin': np.random.choice([0, 1], size=100)
}

df = pd.DataFrame(data)
df.loc[np.random.choice(df.index, size=10), 'Possession'] = np.nan
df.loc[np.random.choice(df.index, size=5), 'ShotsOnTarget'] = np.nan

df.to_csv('data/raw_data.csv', index=False)
print("Created data/raw_data.csv (Arsenal Match Theme)")
