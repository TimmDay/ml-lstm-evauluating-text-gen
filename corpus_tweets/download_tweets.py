import sys
import os
import csv
import re
from auths import api  # api = tweepy.API(auth, wait_on_rate_limit=True)
# import pycountry  # to convert ISO 639.1 (returned from twitter api) to ISO 639.2 (desired)

fileCount = 0  # for counting the number of files read from a directory

def helper_tweep_read_writer(line):
  print('begin tweepy')
  tweet_id = line.strip()
  tweet_split_txt = api.get_status(tweet_id, tweet_mode='extended').full_text
  print(tweet_split_txt)
  tweet_split_txt = tweet_split_txt.split()
  

  treated_sen = []
  for word in tweet_split_txt:
    if (word.startswith('@')):
      pass
    elif (word.startswith('https:')):
      pass
    elif (word.startswith('RT')):
      pass
    elif (word.startswith('#')):
      word = word[1:] # remove the hashtag
      sp_word = re.findall('[A-Z][^A-Z]*', word) # split the word around capital letters (but only if they are not consequtive ie acronym)
      print(sp_word)
      sp_word = ' '.join(sp_word) # join back to string
      # .lower()
      treated_sen.append(sp_word)
    # remove chopped off words that end in '...' TODO?
    else:
      treated_sen.append(word) #.lower()
    
  final = ' '.join(treated_sen) + '\n' # add new line to the end to signal end of tweet
  outFile.write(final)

def helper_process_dir(dir):
    for root, dirs, files in os.walk(arg):
        for file in files:
            print(file)

            if file.endswith(".id"):
                try:
                    f = open(arg + "/" + file, 'r')
                    lines = f.readlines()
                    tweet_lang = file[:3] # take the first three letters of file name as the language
                    print(tweet_lang)

                    for line in lines:
                        try:
                            helper_tweep_read_writer(line)
                        except Exception:
                            pass
                    f.close()

                except IOError:
                    print("could not read a file at: ", arg)

def helper_process_file(arg):
  if arg.endswith(".id"):
    try:
      print('in file function')
      f = open(arg, 'r')
      # print(f)
      lines = f.readlines()
      # print(lines)
      for line in lines:
        try:
          # print(line)
          helper_tweep_read_writer(line)
        except Exception:
          pass
      f.close()
    except IOError:
      print("could not read a file at: ", arg)

print('hello')
# print(sys.argv)




### user dictated process from cmd line
if (len(sys.argv) < 2):
  print('please provide one or more arguments (directories or files) in the form of file paths from the cwd')
  print('only files with the suffix .id will be considered.')
else :
  outFile = open("tweet-corpus_melb_19-4-15.txt","w+")

  for i in range(1, len(sys.argv)):
    arg = os.path.join(os.getcwd(), sys.argv[i])

    if os.path.isdir(arg):  # if command line arg is a path to a DIRECTORY
      print('is dir')
      helper_process_dir(arg)

    elif os.path.isfile(arg):  # if cmd line argument is a path to a FILE
      print('is file')
      helper_process_file(arg)

    print('number of files read:', fileCount)

    ## Command line tests
    # python download-tweets.py train/hin.id train/vie.id
    # python download-tweets.py train