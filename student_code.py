import math
import re

class Bayes_Classifier:

    def __init__(self):
        self.total_training = 0
        self.positive = 0
        self.negative = 0
        self.word_list = {}
        self.stop_words = {'a', 'able', 'about', 'above', 'according', 'accordingly', 'across', 'actually', 'after', 'afterwards', 'again', 'against', 'all', 'allow', 'allows', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'an', 'and', 'another', 'any', 'anybody', 'anyhow', 'anyone', 'anything', 'anyway', 'anyways', 'anywhere', 'apart', 'appear', 'appreciate', 'appropriate', 'are', 'around', 'as', 'aside', 'ask', 'asking', 'associated', 'at', 'available', 'away', 'awfully', 'b', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'behind', 'being', 'believe', 'below', 'beside', 'besides', 'best', 'better', 'between', 'beyond', 'both', 'brief', 'but', 'by', 'c', "c's", 'came', 'can', 'cannot', 'cant', 'cause', 'causes', 'certain', 'certainly', 'changes', 'clearly', 'co', 'com', 'come', 'comes', 'concerning',
'consequently', 'consider', 'considering', 'contain', 'containing', 'contains', 'corresponding', 'could', 'course', 'currently', 'd', 'definitely', 'described', 'despite', 'did', 'different', 'do', 'does', 'doing', 'done', 'down', 'downwards', 'during', 'e', 'each', 'edu', 'eg', 'eight', 'either', 'else', 'elsewhere', 'enough', 'entirely', 'especially', 'et', 'etc', 'even', 'ever', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'ex', 'exactly', 'example', 'except', 'f', 'far', 'few', 'fifth', 'first', 'five', 'followed', 'following', 'follows', 'for', 'former', 'formerly', 'forth', 'four', 'from', 'further', 'furthermore', 'g', 'get', 'gets', 'getting', 'given', 'gives', 'go', 'goes', 'going', 'gone', 'got', 'gotten', 'greetings', 'h', 'had', 'happens', 'hardly', 'has', 'have', 'having', 'he', 'hello', 'help', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'hi', 'him', 'himself', 'his', 'hither', 'hopefully', 'how', 'howbeit', 'however', 'i', "i'll", "i've",
'ie', 'if', 'ignored', 'immediate', 'in', 'inasmuch', 'inc', 'indeed', 'indicate', 'indicated', 'indicates', 'inner', 'insofar', 'instead', 'into', 'inward', 'is', 'it', "it'll", 'its', 'itself', 'j', 'just', 'k', 'keep', 'keeps', 'kept', 'know', 'knows', 'known', 'l', 'last', 'lately', 'later', 'latter', 'latterly', 'least', 'less', 'lest', 'let', 'like', 'liked', 'likely', 'little', 'look', 'looking', 'looks', 'ltd', 'm', 'mainly', 'many', 'may', 'maybe', 'me', 'mean', 'meanwhile', 'merely', 'might', 'more', 'moreover', 'most', 'mostly', 'much', 'must', 'my', 'myself', 'n', 'name', 'namely', 'nd', 'near', 'nearly', 'necessary', 'need', 'needs', 'neither', 'never', 'nevertheless', 'new', 'next', 'nine', 'no', 'nobody', 'non', 'none', 'noone', 'nor', 'normally', 'not', 'nothing', 'novel', 'now', 'nowhere', 'o', 'obviously', 'of', 'off', 'often', 'oh', 'ok', 'okay', 'old', 'on', 'once', 'one', 'ones', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'ought', 'our', 'ours', 'ourselves', 'out', 'outside', 'over', 'overall', 'own', 'p', 'particular', 'particularly', 'per', 'perhaps', 'placed', 'please', 'plus', 'possible', 'presumably', 'probably', 'provides', 'q', 'que', 'quite', 'qv', 'r', 'rather', 'rd', 're', 'really', 'reasonably', 'regarding', 'regardless', 'regards', 'relatively', 'respectively', 'right', 's', 'said', 'same', 'saw', 'say', 'saying', 'says', 'second', 'secondly', 'see', 'seeing', 'seem', 'seemed', 'seeming', 'seems', 'seen', 'self', 'selves', 'sensible', 'sent', 'serious', 'seriously', 'seven', 'several', 'shall', 'she', 'should', 'since', 'six', 'so', 'some', 'somebody', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhat', 'somewhere', 'soon', 'sorry', 'specified', 'specify', 'specifying', 'still', 'sub', 'such', 'sup', 'sure', 't', 'take', 'taken', 'tell', 'tends', 'th', 'than', 'thank', 'thanks', 'thanx', 'that', 'thats', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein', 'theres', 'thereupon', 'these', 'they', "they'll", "they've", 'think', 'third', 'this', 'thorough', 'thoroughly', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'took', 'toward', 'towards', 'tried', 'tries', 'truly', 'try', 'trying', 'twice', 'two', 'u', 'un', 'under', 'unfortunately', 'unless', 'unlikely', 'until', 'unto', 'up', 'upon', 'us', 'use', 'used', 'useful', 'uses', 'using', 'usually', 'uucp', 'v', 'value', 'various', 'very', 'via', 'viz', 'vs', 'w', 'want', 'wants', 'was', 'way', 'we', "we'll", "we've", 'welcome', 'well', 'went', 'were', 'what', 'whatever',
'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'wish', 'with', 'within', 'without', 'would', 'would', 'x', 'y', 'yes', 'yet', 'you', "you'll", "you've", 'your', 'yours', 'yourself', 'yourselves', 'z', 'zero'}
        self.punctuation = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~1234567890 ')
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
                word = self.stemming(word)
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
            pos_poss = 0
            neg_poss = 0
            for word in words:
                pos_poss = pos_poss+self.log_prob_calc(word, "5")
                neg_poss = neg_poss+self.log_prob_calc(word, "1")
                sentiment_result.append("5" if ((pos_score*pos_poss)>(neg_score*neg_poss)) else "1")
        return sentiment_result

    def log_prob_calc(self, word, sentiment):
        anti_sent = ("5" if sentiment == "1" else "1")
        if (word, sentiment) in self.word_list.keys() and (word, anti_sent) in self.word_list.keys():
            word_count = self.word_list[(word,sentiment)]
            if sentiment == "5":
                return math.log((word_count+1)/self.positive)
            else:
                return math.log((word_count+1)/self.negative)
        else:
            return 0

    def stemming(self, word):
        length = len(word)
        if length > 3 and (word[length-1] == 's' or word[length-1] == 'y'):
            word = word[:(length-1)]
        length = len(word)
        if word[(length-2):(length)] in {"ed", "es", "er", "al", "ic", "ou", "le"} and length > 5:
            word = word[:(length-2)]
        elif length>5 and word[(length-3):length] in {"ing", "ion", "ive", "ism", "ant", "ent", "ate", "ize", "ive", "est", "ous"}:
            word = word[:(length-3)]
        elif "haha" in word:
            word = "haha"

        return word
