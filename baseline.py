# mp4.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created Fall 2018: Margaret Fleck, Renxuan Wang, Tiantian Fang, Edward Huang (adapted from a U. Penn assignment)
# Modified Spring 2020: Jialu Li, Guannan Guo, and Kiran Ramnath
# Modified Fall 2020: Amnon Attali, Jatin Arora
# Modified Spring 2021 by Kiran Ramnath
"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

def baseline(train, test):
        '''
        input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
        output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
        '''

        # for each word w, count how many times w occurs with each tag in the training data
        # give w the tag that was seen most often
        # for unseen words, guess the tag that's seen most often in training dataset

        word_count = {}
        tag_count = {}

        for sentence in train:
                for word, tag in sentence:
                        
                        # increment word_count by 1 for the specific tag
                        if word not in word_count:
                                word_count[word] = { tag: 1 }
                        else: 
                                curr_count = word_count[word].get(tag, 0) + 1
                                word_count[word][tag] = curr_count

                        # update the total count of the tag
                        curr_tag_count = tag_count.get(tag, 0) + 1
                        tag_count[tag] = curr_tag_count

        max_tag_type = max(tag_count, key=tag_count.get)


        answer = []
        for sentence in test:
                
                sentence_word_tags = []
                for word in sentence:
                        val = (word, max_tag_type) # unseen gets max_tag_type
                        if word in word_count:
                                best_tag = max(word_count[word], key = word_count[word].get)
                                val = (word, best_tag) # otherwise, most often tag for the word
                        sentence_word_tags.append(val)
                answer.append(sentence_word_tags)

        return answer
