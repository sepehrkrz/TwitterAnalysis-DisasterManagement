# Twitter API: Search for 2018 California Camp Fire Tweets and Save as CSV

This code snippet uses the Tweepy library in Python to search for tweets related to the 2018 California Camp Fire and saves the results as a CSV file using the Pandas library.

## Prerequisites

Before you can run this code, you'll need to set up a few things:

1. **Twitter API credentials:** You'll need to create a Twitter developer account and obtain your API credentials, including your `consumer_key`, `consumer_secret`, `access_token`, and `access_token_secret`. 

2. **Python environment:** You'll need to have Python 3 installed on your computer, along with the following libraries:
   - Tweepy: `pip install tweepy`
   - Pandas: `pip install pandas`

## How to Use

1. Copy the code snippet to a Python file, e.g. `campfire_tweets.py`.

2. Replace the placeholders for your Twitter API credentials:

```python
consumer_key = "your_consumer_key"
consumer_secret = "your_consumer_secret"
access_token = "your_access_token"
access_token_secret = "your_access_token_secret"
```

## Parameters
The script has several parameters that you can modify to customize the search:

query: A string containing the search query, e.g. "2018 California Camp Fire" or "#campfire OR #californiawildfire".
lang: The language of the tweets to search for, e.g. "en" for English, "es" for Spanish.
since_id: The date to start searching from, in the format "YYYY-MM-DD".
until: The date to stop searching at, in the format "YYYY-MM-DD".
tweet_mode: The mode to retrieve tweets in, either "extended" for full text or "compat" for truncated text.
You can modify these parameters by changing the corresponding variables in the code:

```python
query = "#campfire OR #californiawildfire OR #buttecounty OR #paradiseca OR #magalia OR #chico OR #campfire2018 OR #woolseyfire OR #hillfire"
lang = "en"
since_id = "2018-11-08"
until = "2018-12-07"
tweet_mode = "extended"
```
