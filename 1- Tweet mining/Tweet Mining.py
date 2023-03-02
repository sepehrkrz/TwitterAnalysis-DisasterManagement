import tweepy
import pandas as pd

# Twitter API credentials
consumer_key = "your_consumer_key"
consumer_secret = "your_consumer_secret"
access_token = "your_access_token"
access_token_secret = "your_access_token_secret"

# Authenticate with the Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# Create a Tweepy API object
api = tweepy.API(auth)

# Define the search query
query = "#campfire OR #californiawildfire OR #buttecounty OR #paradiseca OR #magalia OR #chico OR #campfire2018 OR #woolseyfire OR #hillfire"

# Define the date range of interest
start_date = "2018-11-08"
end_date = "2018-12-07"

# Use the Tweepy Cursor object to search for tweets matching the query and date range
tweets = tweepy.Cursor(api.search_tweets,
                       q=query,
                       lang="en",
                       since_id=start_date,
                       until=end_date,
                       tweet_mode="extended").items()

# Create a list of dictionaries to store the data
data = []
for tweet in tweets:
    data.append({
        'id': tweet.id_str,
        'created_at': tweet.created_at,
        'full_text': tweet.full_text,
        'user_screen_name': tweet.user.screen_name,
        'user_location': tweet.user.location,
        'retweet_count': tweet.retweet_count,
        'favorite_count': tweet.favorite_count,
    })

# Convert the list of dictionaries to a Pandas DataFrame
df = pd.DataFrame(data)

# Save the DataFrame as a CSV file
df.to_csv('campfire_tweets.csv', index=False)
