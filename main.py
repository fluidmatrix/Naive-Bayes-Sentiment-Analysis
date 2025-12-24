from nltk.corpus import twitter_samples # type: ignore
from helper import count_tweets, train_naive_bayes, naive_bayes_predict, test_naive_bayes
import numpy as np 

# Get the set of Positive and Negative Tweets
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

# Split the data in Train and Validation Sets
train_pos = all_positive_tweets[:4000]
test_pos =  all_positive_tweets[4000:]
train_neg = all_negative_tweets[:4000]
test_neg =  all_negative_tweets[4000:]

# Combine the test sets
train_x = train_pos + train_neg
test_x = test_pos + test_neg

# avoid assumptions about the length of all_positive_tweets
train_y = np.append(np.ones(len(train_pos)), np.zeros(len(train_neg)))
test_y = np.append(np.ones(len(test_pos)), np.zeros(len(test_neg)))


# Build the freqs dictionary for later uses
freqs = count_tweets({}, train_x, train_y)

logprior, loglikelihood = train_naive_bayes(freqs, train_x, train_y)

# Test with your own tweet - feel free to modify `my_tweet`
my_tweet = 'Neutral'

p = naive_bayes_predict(my_tweet, logprior, loglikelihood)

if(p > 0):
    print('\033[92mThe Tweet is POSITIVE')
elif(p < 0):
    print('\033[91mThe Tweet is NEGATIVE')
else:
    print('\033[90mThe Tweet is Neutral')





