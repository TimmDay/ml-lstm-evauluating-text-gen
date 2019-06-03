"""
twitter crawler program to build english tweet corpus
Author: Tim Day
"""
import tweepy
import json
from auths import api

# gather a large sample of full-length tweets from 100 km radius of London
London_en_tweets = tweepy.Cursor(api.search, geocode="-37.8208,144.9887,100km", tweet_mode='extended', lang="en").items(2000)

# function to append the tweets to a text file, 
# store json data in another file and print the yield rate to the console
def tweet_to_files(filetxt, filejson):
    print('filtering out short tweets')
    with open(filetxt, 'w', encoding="utf-8") as outtxt, open(filejson, 'w') as outjson:
        count = 0
        total = 0
        jsondata = {}
        jsondata['tweets'] = []

        for tweet in London_en_tweets:
            content = tweet.full_text

            if len(content) >= 50 and count < 10000:
                outtxt.write(tweet.id_str + "\n") # store ids for privacy reg sharing
                jsondata['tweets'].append(tweet._json)
                count += 1
            total += 1
            print('running yield rate: ', count / total)
            print('tweets so far /10000', count)

            if count == 10000:
                json.dump(jsondata, outjson)  # json file only populates at the end
                print('Final passed filter check, expect 10000: ', count)
                print('Final Yield Rate: ', count / total)

if __name__ == "__main__":
    tweet_to_files("tweets_melb.id","tweets_melb.json")
