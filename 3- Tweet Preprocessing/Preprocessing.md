## Text Preprocessing for Campfire States
This code performs text preprocessing for tweets related to the 2018 California Campfire wildfire. The dataset used for this code is campfire_states.csv.

### Dependencies
This code requires the following dependencies to be installed:

- pandas
- re
- string
- nltk
- joblib
- In addition, the stopwords corpus from the nltk package must be downloaded.

``` python
import pandas as pd
import re
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from joblib import Parallel, delayed
```

## Data Loading and Preprocessing
First, the dataset is loaded using pandas. The "Text" column is converted to a string data type, and a new column "Raw_Text" is created to store the original, unprocessed text
```python
train = pd.read_csv('campfire_states.csv')
train["Text"] = train["Text"].map(str)
train["Raw_Text"]= train["Text"]
```
Next, several functions are defined to perform specific text cleaning tasks. These functions include removing URLs, HTML tags, emojis, punctuation, mentions, hashtags, and numbers. In addition, a function is defined to remove extra characters that are not part of the ASCII character set.
```python
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
    text = re

```
