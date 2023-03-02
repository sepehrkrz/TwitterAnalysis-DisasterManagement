import re
import pandas as pd
from joblib import Parallel, delayed
import joblib
import time, math

# load data
Twitter_Dataset = pd.read_csv('campfire.csv')
states = pd.read_csv('states.csv')

# preprocess location data for efficient matching
states['State'] = states['State'].str.upper()
states['Abbreviation'] = states['Abbreviation'].str.upper()
Twitter_Dataset['author.location'] = Twitter_Dataset['author.location'].str.upper()

# create a dictionary to store state-specific checks
state_checks = {
    'Kansas': ['MISSOURI', ', MO', ',  MO', 'ARKANSAS', ',MO'],
    'Delaware': [', PA', ' DELAWARE COUNTY', ', MO'],
    'Maine': [', OH', ' OHIO', ',  MO', ', OH'],
    'Washington': [', DC', ', D.C.', ',D.C', ' DISTRICT OF COLUMBIA', ' DC', ' D.C.', ' DISTRICT OF COLUMBIA', ' DC', ', DC', 'D.', ', MO', ', UT', ', PA', ', MD'],
}

# Replace NaN values in 'author.location' column with an empty string
Twitter_Dataset['author.location'].fillna('', inplace=True)

# filter Twitter_Dataset based on state or abbreviation match
filtered_tweets = Twitter_Dataset[
    Twitter_Dataset['author.location'].str.contains(r'\b(' + '|'.join(states['State']) + r'|' + '|'.join(states['Abbreviation']) + r')\b')
]

# create a new DataFrame for the California campfire
df1 = pd.DataFrame(columns=['Date', 'Location', 'ID', 'Text', 'State', 'Author'])

# Extracting desired location (Parallalized using Joblib)
# iterate through filtered_tweets to check each state's criteria
def check_state_criteria(row, states, state_checks):
    loc = row['author.location']
    for index_states, row_states in states.iterrows():
        state = row_states['State']
        if state in state_checks:
            if any(check in loc for check in state_checks[state]):
                continue
        if state in loc or row_states['Abbreviation'] in loc:
            return {
                'Date': row['created_at'],
                'Location': row['author.location'],
                'ID': row['id'],
                'Text': row['text'],
                'State': row_states['State'],
                'Author': row['author.username']
            }
    return None
start = time.time()
results = joblib.Parallel(n_jobs=-1)(
    joblib.delayed(check_state_criteria)(row, states, state_checks) for index, row in filtered_tweets.iterrows())
end = time.time()
print('Run time with Parallelization:')
print('{:.4f} s'.format(end-start))


# write results to file
df1.to_csv('campfire_sates.csv', index=False)