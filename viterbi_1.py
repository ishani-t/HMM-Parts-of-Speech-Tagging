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
# Modified Spring 2021 by Kiran Ramnath (kiranr2@illinois.edu)

"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""
import math

def viterbi_1(train, test):
        '''
        input:  training data (list of sentences, with tags on the words)
                test data (list of sentences, no tags on the words)
        output: list of sentences with tags on the words
                E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
        '''

        # Count occurrences of tags, tag pairs, tag/word pairs.
        tags, tag_pairs, tag_word_pairs = training_occurences(train)

        # Compute smoothed probabilities
        # Take the log of each probability
        # use formulas from lec24 slide 26
        k = 1e-6
        all_tags = list(tags.keys())
        train_len = len(train)
        V = len(tag_word_pairs)

        initial_probs = calc_initial_probs(tags, k, train, train_len)
        transition_probs = calc_transition_probs(tags, tag_pairs, k)
        emission_probs = calc_emission_probs(tags, tag_word_pairs, k, V)
        


        # Construct the trellis. 
        # Notice that for each tag/time pair, you must store not only the probability of the best path but also a pointer to the previous tag/time pair in that path.
        answers = []
        for sentence in test:
                t = []
                to_return = []
                
                for index, word in enumerate(sentence):
                        nodes = []

                        if index == 0: # word = START, calculate initial node probs
                                for tag in all_tags:
                                        emission_prob = math.log(k) - math.log(tags[tag] + k * (V + 1))
                                        if word in emission_probs:
                                                emission_prob = emission_probs[word].get(tag, emission_prob)
                                        
                                        # initial node probability: v_{j,1} = pi_j * b_{j, x_1}
                                        node_prob = emission_prob + initial_probs[tag]
                                        nodes.append( (node_prob, tag) )
                                t.append(nodes)
                        else: # calculate node probabilities, find max of each layer to tag the word
                                for tag in all_tags:                                          
                                        backpointer = all_tags[0]
                                        node_probability = float('-inf')

                                        for i, prev_tag in enumerate(all_tags):  # max over tags from the previous layer of trellis

                                                emission_prob = math.log(k) - math.log(tags[tag] + k * (V + 1))
                                                if word in emission_probs:
                                                        emission_prob = emission_probs[word].get(tag, emission_prob)
                                                
                                                back_prob = t[index - 1][i][0]                                          # v_{i, t-1}: need to access previous layer of trellis and max over it
                                                edge_prob = transition_probs[prev_tag][tag] + emission_prob             # edge probability: e_{i,j,t} = a_{i,j} * b_{j, x_t}
                                                node_prob = back_prob + edge_prob                                       # node probability: v_{j,t} = \max_{i} v_{i, t-1} * e_{i,j,t}
                                                
                                                if node_prob > node_probability:
                                                        node_probability = node_prob
                                                        backpointer = prev_tag
                                                
                                        nodes.append( (node_probability, backpointer) )
                                t.append(nodes)
                                
                                # Return the best path through the trellis.
                                # for each word (layer) in the trellis, find the max probability of that tag, then proceed!
                                max_node_prob = float('-inf')
                                back_tag = 'START'
                                for prob, tag in nodes:
                                        if prob > max_node_prob:
                                                back_tag = tag
                                                max_node_prob = prob
                                to_return.append((sentence[index - 1], back_tag))
        
                to_return.append(('END','END'))
                answers.append(to_return)

        return answers

def training_occurences(train):
        tags = {}
        tag_pairs = {}
        tag_word_pairs = {}

        for sentence in train:
                pair_t = 'START' # keep track of previous tag for pairs

                for w, t in sentence:

                        # update tags
                        curr_tag_count = tags.get(t, 0) + 1
                        tags[t] = curr_tag_count
                        
                        # update tag_pairs
                        pair = (pair_t, t)
                        curr_pair_count = tag_pairs.get(pair, 0) + 1
                        tag_pairs[pair] = curr_pair_count
                        
                        # update tag_word_pairs
                        if w not in tag_word_pairs: # if the word isn't in the dict yet
                                tag_word_pairs[w] = {t : 1}
                        else: # increment tag for that word
                                curr_tag_word_pair = tag_word_pairs[w].get(t, 0) + 1
                                tag_word_pairs[w][t] = curr_tag_word_pair
                                
                        pair_t = t
                
        return tags, tag_pairs, tag_word_pairs

def calc_initial_probs(tags, k, train, train_len):
        # Initial probabilities (How often does each tag occur at the start of a sentence?)
        initial_probs = {}
        all_tags = list(tags.keys())
        N = len(all_tags)

        for t in all_tags:
                initial_probs[t] = 0
        for sentence in train:
                tag = sentence[1][1]
                curr_tag_val = initial_probs.get(tag, 0) + 1
                initial_probs[tag] = curr_tag_val
        for t in all_tags:
                prob = (initial_probs.get(t) + k) / (train_len + k * N)
                log_prob = math.log(prob)
                initial_probs[t] = log_prob
        return initial_probs

def calc_transition_probs(tags, tag_pairs, k):
        # Transition probabilities (How often does tag tb follow tag ta?)
        transition_probs = {}
        all_tags = list(tags.keys())
        N = len(all_tags)
        
        for ta in all_tags:
                tag_ta_dict = {}
                num_tas = tags[ta]
                for tb in all_tags:
                        prob = (tag_pairs.get((ta, tb), 0) + k) / (num_tas + k * N)
                        log_prob = math.log(prob)
                        tag_ta_dict[tb] = log_prob
                transition_probs[ta] = tag_ta_dict
        return transition_probs

def calc_emission_probs(tags, tag_word_pairs, k, V):
        # Emission probabilities (How often does tag t yield word w?)
        emission_probs = tag_word_pairs
        for w in tag_word_pairs:
                for t, num in tag_word_pairs[w].items():
                        prob = (num + k) / (tags[t] + k * (V + 1))
                        log_prob = math.log(prob)
                        emission_probs[w][t] = log_prob
        return emission_probs
