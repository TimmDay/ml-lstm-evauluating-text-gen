# file = open('corpus_wiki/2019-04-08_wiki_corpus.txt', mode='r')
# file = open('corpus_tweets/tweet-corpus_melb_19-4-15.txt', mode='r')

# file = open('corpus_tweets/2019-04-04_tweet-corpus.txt', mode='r')
# text = file.read().lower()
# print('corpus length: ', len(text))

with open('corpus_wiki/2018-04-08_wiki_corpus.txt', mode='r') as infile:
  lines=0
  words=0
  characters=0
  for line in infile:
    wordslist=line.split()
    lines=lines+1
    words=words+len(wordslist)
    characters += sum(len(word) for word in wordslist)
print(lines)
print(words)
print(characters)