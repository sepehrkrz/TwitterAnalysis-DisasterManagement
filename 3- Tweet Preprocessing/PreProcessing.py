import pandas as pd
import re
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from joblib import Parallel, delayed


train = pd.read_csv('campfire_states.csv')
train["Text"] = train["Text"].map(str)
train["Raw_Text"]= train["Text"]

def remove_URL(text):
    url = re.compile(r"https?://\S+|www\.\S+")
    return url.sub(r"", text)

def remove_html(text):
    html = re.compile(r"<.*?>")
    return html.sub(r"", text)

def remove_emoji(string):
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", string)

def remove_punct(text):
    table = str.maketrans("", "", string.punctuation)
    return text.translate(table)

def text_cleaning(text):
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'@[A-Za-z0-9]+', '', str(text))     # removing @mentions
    text = re.sub(r'@[A-Za-zA-Z0-9]+', '', str(text))  # removing @mentions 
    text = re.sub(r'@[A-Za-z]+', '', str(text))        # removing @mentions
    text = re.sub(r'@[-)]+', '', str(text))            # removing @mentions
    text = re.sub(r'#', '', str(text) )                # removing '#' sign
    text = re.sub(r'RT[\s]+', '', str(text))           # removing RT
    text = re.sub(r'https?\/\/\S+', '', str(text))     # removing the hyper link
    text = re.sub(r'&[a-z;]+', '', str(text))          # removing '&gt;'
    text = re.sub(r'[0-9]', '', str(text))             # remove numbers
    return text

def remove_extra_characters(text):
    #text = re.sub("[^a-zA-Z0-9]+", " ",text)
    text = re.sub(r'[^\x00-\x7f]', "", text)
    return text

stop = set(stopwords.words("english"))

def remove_stopwords(text):
    text = [word.lower() for word in text.split() if word.lower() not in stop]
    return " ".join(text)

def preprocess_text(text):
    text = text_cleaning(text)
    text = remove_URL(text)
    text = remove_html(text)
    text = remove_emoji(text)    
    text = remove_punct(text)
    text = text.lower()
    text = remove_extra_characters(text)
    text_stop = remove_stopwords(text)
    text_stop = remove_punct(text_stop) 
    text_stop = remove_emoji(text_stop)
    return text_stop

#Drop retweets
df = train[train["Raw_Text"].str.contains("RT ") == False]

# Parallelize the preprocessing step using joblib
num_cores = -1
preprocessed_texts = Parallel(n_jobs=num_cores)(
    delayed(preprocess_text)(text) for text in df["Raw_Text"]
)

df["Text_Stop"] = preprocessed_texts
df.to_csv("Campfire_State_Preprocess.csv", index=False)
df.head()