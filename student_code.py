import math
import re

class Bayes_Classifier:

    def __init__(self):
        self.total_training = 0
        self.positive = 0
        self.negative = 0
        self.word_list = {}
        self.stop_words = { "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", ", ", ". ", "\'", "\' ", " \'", "-", "+", "=","re", "ve", "s", " " }
        self.punctuation = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~1234567890')
        return


    #param {lines}
    #return none
    def train(self, lines):
        self.total_training = len(lines)
        for line in lines:
            result_list = line.split('|')
            #registering the ratings
            if result_list[0] == "1": 
                self.negative += 1
            else: 
                self.positive += 1
            #registering the words
            for word in re.split('(\W+)', result_list[2]):
                word = word.lower()
                word = word.strip()
                if word not in self.stop_words and len([char for char in list(word) if char in self.punctuation]) == 0 and word != '':
                    if (word,result_list[0]) in self.word_list.keys():
                        self.word_list[(word,result_list[0])] += 1
                    else:
                        self.word_list[(word,result_list[0])] = 1 
        word_list = self.word_list   
        return word_list


    #param{list of strings}
    #return listof(review:"1"/"5")
    def classify(self, lines):
        sentiment_result = []
        for line in lines:
            result_list = line.split('|')
            #registering the words
            words = re.split('(\W+)', result_list[2])
            pos_score = self.positive/self.total_training
            neg_score = self.negative/self.total_training
            for word in words:
                pos_score = pos_score*self.log_prob_calc(word, "5")
                neg_score = neg_score*self.log_prob_calc(word, "1")
                sentiment_result.append("5" if (pos_score>neg_score) else "1")
        return sentiment_result

    def log_prob_calc(self, word, sentiment):
        if (word, sentiment) in self.word_list.keys():
            word_count = self.word_list[(word,sentiment)]
            if sentiment == "5":
                return math.log((word_count+1)/self.positive)
            else:
                return math.log((word_count+1)/self.negative)
        else:
            return 1
    
    