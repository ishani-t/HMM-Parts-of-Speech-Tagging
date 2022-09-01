# HMM-Parts-of-Speech-Tagging
# CS440 SP21

- baseline.py
- viterbi_1.py
- viterbi_2.py

1. Baseline tagger: For each word w, it counts how many times w occurs with each tag in the training data. When processing the test data, it consistently gives w the tag that was seen most often. For unseen words, it guesses the tag that's seen the most often in training dataset.
2. Viterbi
3. Viterbi-2

 `python3 mp4.py --train data/brown-training.txt --test data/brown-dev.txt --algorithm [baseline, viterbi_1, viterbi_2, viterbi_ec]`
