import pandas as pd

def to_sentiment(rating):
    rating = int(rating)
    if rating <= 2:
        return 0
    elif rating == 3:
        return 1
    else: 
        return 2

def get_data(filename):
    data = pd.read_csv(filename)
    data['sentiment'] = data.score.apply(to_sentiment)

    return data