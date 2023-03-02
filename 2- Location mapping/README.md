# Filtering Tweets Based on User Location and Mapping Them to States in the United States

This code is designed to filter mined Tweets based on user location and map them to states in the United States. The code reads in data from a csv file ("campfire.csv"), which contains Twitter data related to the 2018 California Camp Fire. Additionally, the code requires a separate csv file ("states.csv") containing information about the states in the United States.

## Required Libraries

This code requires the following libraries:
- re
- pandas
- joblib
- time
- math

## Loading Data

The code loads data from the "campfire.csv" file using the pandas library. It also loads a file named "states.csv" which contains information about the states in the United States.

```python
Twitter_Dataset = pd.read_csv('campfire.csv')
states = pd.read_csv('states.csv')
```
## Preprocessing Location Data
To efficiently match Twitter users to states based on their location, the location data in both datasets needs to be preprocessed. The code converts all state names and abbreviations in the "states" DataFrame to uppercase. It also converts all location data in the "Twitter_Dataset" DataFrame to uppercase.

```python
states['State'] = states['State'].str.upper()
states['Abbreviation'] = states['Abbreviation'].str.upper()
Twitter_Dataset['author.location'] = Twitter_Dataset['author.location'].str.upper()
```

## State-Specific Checks
Some states have specific requirements for matching users to a state. For example, tweets from users located in Kansas might mention Missouri or Arkansas in their location data, and we still want to count them as being located in Kansas. To handle this, the code creates a dictionary called "state_checks" that holds these state-specific requirements.

```python
state_checks = {
    'Kansas': ['MISSOURI', ', MO', ',  MO', 'ARKANSAS', ',MO'],
    'Delaware': [', PA', ' DELAWARE COUNTY', ', MO'],
    'Maine': [', OH', ' OHIO', ',  MO', ', OH'],
    'Washington': [', DC', ', D.C.', ',D.C', ' DISTRICT OF COLUMBIA', ' DC', ' D.C.', ' DISTRICT OF COLUMBIA', ' DC', ', DC', 'D.', ', MO', ', UT', ', PA', ', MD'],
}

```

## Filtering Data
The code first replaces any NaN values in the "author.location" column with an empty string. It then filters the "Twitter_Dataset" based on state or abbreviation match using the regular expression pattern below. This code uses the '|' (pipe) symbol to match either the state name or its abbreviation.
```python
filtered_tweets = Twitter_Dataset[
    Twitter_Dataset['author.location'].str.contains(r'\b(' + '|'.join(states['State']) + r'|' + '|'.join(states['Abbreviation']) + r')\b')
]
```

## Extracting Desired Location
The code creates a new DataFrame called "df1" to store the filtered data that matches the state-specific criteria. The function "check_state_criteria" is defined to check each state's criteria and extract the desired location information. This function is parallelized using the "joblib" library to speed up the computation.
```python
def check_state_criteria(row, states, state_checks):
    # ...
    
start = time.time()
results = joblib.Parallel(n_jobs=-1)(
    joblib.delayed(check_state_criteria)(row, states, state_checks) for index, row in filtered_tweets.iterrows())
end = time.time()
```
## Preprocessing User Location Data and Filtering Mined Tweets Based on Location
This Python code filters mined tweets based on user location and maps them to states in the United States. The code uses a CSV file named states.csv that holds information about states.
```python
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
```
The code performs the following steps:

Load the Twitter dataset (campfire.csv) and the states information (states.csv) into Pandas data frames.
1- Preprocess the location data for efficient matching by converting all state names and abbreviations to uppercase.
2- Create a dictionary called state_checks that holds state-specific matching criteria to check for false matches. The 
3- criteria were chosen manually by analyzing the user location data.
4- Replace NaN
