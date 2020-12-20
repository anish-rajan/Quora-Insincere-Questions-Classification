!pip install pandarallel

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
import string
import spacy
import os
import multiprocessing
import gc
import io

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from pandarallel import pandarallel

from nltk import word_tokenize,pos_tag
from nltk.corpus import stopwords

from collections import Counter
from tqdm import tqdm
from wordcloud import WordCloud
from keras.preprocessing.text import Tokenizer
from nltk.tokenize import TweetTokenizer

tqdm.pandas()
cores = multiprocessing.cpu_count()

pandarallel.initialize()
tqdm(pandarallel)

class data_visualization() :
	def __init__ (self) :
		self.train = pd.read_csv("../input/quora/train.csv")
		self.test = pd.read_csv("../input/quora/test.csv")

	def sincere_insincere_count_distribution(self) :
		fig, ax = plt.subplots()
		plt.title('Distribution of Questions', fontsize = 30)
		g = sns.countplot(self.train['target'], palette = 'viridis')
		g.set_xticklabels(['Sincere', 'Insincere'])
		plt.show()

	def number_of_sentences_per_question(self) :
		copy_data = self.train.sample(n = 10000)
		nlp = spacy.load('en')
		copy_data['tokens'] = [nlp(text, disable = ['ner', 'tagger','textcat']) for text in copy_data['question_text']]
		copy_data['num_tokens'] = [len(token) for token in copy_data['tokens']]
		sentences = [list(i.sents) for i in copy_data['tokens']]
		copy_data['num_of_sentence'] = [len(i) for i in sentences]
		# Linear Scale
		fig, ax = plt.subplots()
		g = sns.countplot(copy_data['num_of_sentence'], hue = copy_data['target'])
		plt.title('Number of Sentences per Question', fontsize = 20)
		ax.set(yscale = 'linear')
		plt.show()

		#Logarithm Scale
		fig, ax = plt.subplots()
		g = sns.countplot(copy_data['num_of_sentence'], hue = copy_data['target'])
		plt.title('Number of Sentences per Question', fontsize = 20)
		ax.set(yscale = 'log')
		plt.show()

	def word_cloud(self) :
		copy_data = self.train.sample(n = 10000)
		text = ' '.join(copy_data[copy_data['target'] == 0].question_text.str.lower().values[-1000000:])
		wordcloud = WordCloud(max_font_size = None, background_color = 'black', width = 1200, height = 1000).generate(text)
		plt.figure(figsize = (12, 8))
		plt.imshow(wordcloud)
		plt.title('Top words in question text')
		plt.axis("off")
		plt.show()
		text = ' '.join(copy_data[copy_data['target'] == 1].question_text.str.lower().values[-1000000:])
		wordcloud = WordCloud(max_font_size = None, background_color = 'black', width = 1200, height = 1000).generate(text)
		plt.figure(figsize = (12, 8))
		plt.imshow(wordcloud)
		plt.title('Top words in question text')
		plt.axis("off")
		plt.show()

	def question_len_in_characters(self) :
		avg_len_char_in_train_set = format(np.mean(self.train['question_text'].apply(lambda y: len(y))))
		print("Average character length of questions in train data set is", avg_len_char_in_train_set)
		avg_len_in_train_set = format(np.mean(self.train['question_text'].apply(lambda y: len(y.split()))))
		print("Average length of each question in train data set is", avg_len_in_train_set)
		max_len_in_train_set = format(np.max(self.train['question_text'].apply(lambda y: len(y.split()))))
		print("Max word length of questions in train data set is", max_len_in_train_set)
		print("We can observe that they are quite a long questions in training set compared to test set. Also average length of training set is same as test set")
		plt.yscale('linear');
		plt.title('Distribution of question length in characters in training set')
		self.train['question_text'].apply(lambda y: len(y.split())).plot(kind = 'hist');
		print("We can observe that most of the word are of lenth 40 or shorter.")
		
	def distribution_of_num_of_words(self) :
		self.train["num_words"] = self.train["question_text"].apply(lambda y: len(str(y).split()))
		plt.figure(figsize=(10, 10))
		sns.violinplot(data = self.train["num_words"])
		plt.title('Distribution of number of words in the training set')
		plt.show()

	def distribution_of_num_of_unique_words(self) :
		self.train["num_unique_words"] = self.train["question_text"].apply(lambda y: len(set(str(y).split())))
		plt.figure(figsize=(10, 10))
		sns.violinplot(data = self.train["num_unique_words"])
		plt.title('Distribution of number of unique words in the training set')
		plt.show()

	def distribution_of_num_of_characters(self) :
		self.train["num_chars"] = self.train["question_text"].apply(lambda y: len(str(y)))
		plt.figure(figsize=(10, 10))
		sns.violinplot(data = self.train["num_chars"])
		plt.title('Distribution of number of characters in the training set')
		plt.show()

	def distribution_of_num_of_punctuations(self) :
		self.train["num_punctuations"] = self.train['question_text'].apply(lambda y: len([x for x in str(y) if x in string.punctuation]))
		plt.figure(figsize = (10, 10))
		plt.ylim(0, 20)
		sns.violinplot(data = self.train["num_punctuations"])
		plt.title('Distribution of number of punctuations in the training set')
		plt.show()
		
	def distribution_of_num_of_words_in_uppercase(self) :
		self.train["num_words_in_upper_case"] = self.train["question_text"].apply(lambda y: len([x for x in str(y).split() if x.isupper()]))
		plt.figure(figsize = (10, 10))
		plt.ylim(0, 10)
		sns.violinplot(data = self.train["num_words_in_upper_case"])
		plt.title('Distribution of number of words in uppercase in the training set')
		plt.show()

	def distribution_of_num_of_words_in_lowercase(self) :
		self.train["num_words_in_lower_case"] = self.train["question_text"].apply(lambda y: len([x for x in str(y).split() if x.islower()]))
		plt.figure(figsize = (10, 10))
		sns.violinplot(data = self.train["num_words_in_lower_case"])
		plt.title('Distribution of number of words in lowercase in the training set')
		plt.show()

	def distribution_of_num_of_stopwords(self) :
		stopwords_in_english = set(stopwords.words("english"))
		self.train["stopwords"] = self.train["question_text"].apply(lambda y: len([x for x in str(y).lower().split() if x in stopwords_in_english]))
		plt.figure(figsize=(10, 10))
		sns.violinplot(data = self.train["stopwords"])
		plt.title('Distribution of number of stopwords in the training set')
		plt.show()

	def distribution_of_mean_word_length(self) :
		self.train["mean_word_length_each_que"] = self.train["question_text"].apply(lambda y: np.mean([len(x) for x in str(y).split()]))
		plt.figure(figsize = (10, 10))
		sns.violinplot(data = self.train["mean_word_length_each_que"])
		plt.title('Distribution of mean word length in each question in the training set')
		plt.show()

	def distribution_of_num_of_title(self) :
		self.train["title"] = self.train["question_text"].apply(lambda y: len([x for x in str(y).split() if x.istitle()]))
		plt.figure(figsize = (10, 10))
		sns.violinplot(data = self.train["title"])
		plt.title('Distribution of number of title in each question in the training set')
		plt.show()

	def show(self) :
		self.sincere_insincere_count_distribution()
		self.number_of_sentences_per_question()
		self.word_cloud()
		self.question_len_in_characters()
		self.test['question_text'].fillna('missing', inplace = True)
		self.train['question_text'].fillna('missing', inplace = True)
		self.distribution_of_num_of_words()
		self.distribution_of_num_of_unique_words()
		self.distribution_of_num_of_characters()
		self.distribution_of_num_of_punctuations()
		self.distribution_of_num_of_words_in_uppercase()
		self.distribution_of_num_of_words_in_lowercase()
		self.distribution_of_num_of_stopwords()
		self.distribution_of_mean_word_length()
		self.distribution_of_num_of_title()


class Vt() :
    def __init__(self, estimators) :
        self.estimators = estimators
        
    def fit(self, train, target) :
        for model in self.estimators:
            model[1].fit(train, target)
    
    def predict_proba(self, test, weights):
        predictions = np.array([0.0]*len(test))
        weights = np.array(weights)
        weights /= np.sum(weights)
        
        for i in range(weights.shape[0]):
            predictions += weights[i]*self.estimators[i][1].predict_proba(test)[ : , 1]
        return predictions

    
class pre_processing() :
    def __init__(self) :
        self.train = pd.read_csv("../input/quora/train.csv")
        self.test = pd.read_csv("../input/quora/test.csv")
        self.qid = self.test.qid
        self.y = self.train.target
        self.train.drop(['target'], axis = 1, inplace = True)
        self.sgd = SGDClassifier(alpha = 0.0001, max_iter = 1000, penalty = "l2", loss = 'modified_huber', verbose = 5, random_state = 2020)
        self.logit = LogisticRegression(penalty = 'l2', max_iter = 200, fit_intercept = True, solver = 'liblinear', multi_class = 'ovr', random_state = 2020)
        self.nb = BernoulliNB(binarize = 0.2)
        self.rfc = RandomForestClassifier(verbose = 5, criterion = 'entropy', max_depth = 100, max_features = 5000, n_estimators = 20, n_jobs = cores , 
                                          min_samples_leaf = 3, min_samples_split = 10)
        self.vt = Vt(estimators = [('sgd', self.sgd), ('logit', self.logit), ('bn', self.nb), ('rfc', self.rfc)])
        self.final_y = None
        
    def data_cleaning(self) :
        puncts = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

        punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }

        def clean_text(x):
            x = str(x)

            for punct in punct_mapping:
                x = x.replace(punct, punct_mapping[punct])

            for punct in puncts:
                x = x.replace(punct, f' {punct} ')

            return x

        def clean_numbers(x):
            x = re.sub('[0-9]{5,}', '#####', x)
            x = re.sub('[0-9]{4}', '####', x)
            x = re.sub('[0-9]{3}', '###', x)
            x = re.sub('[0-9]{2}', '##', x)
            return x

        mispell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization', 'pokémon': 'pokemon',"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}

        def _get_mispell(mispell_dict):
            mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
            return mispell_dict, mispell_re

        mispellings, mispellings_re = _get_mispell(mispell_dict)
        
        def replace_typical_misspell(text):
            def replace(match):
                return mispellings[match.group(0)]
            return mispellings_re.sub(replace, text)

        def text_cleaning(text):
            text = replace_typical_misspell(text)
            text = clean_text(text)
            text = clean_numbers(text)
            return text

        def text_clean_wrapper(curr):
            curr["question_text"] = curr["question_text"].parallel_apply(text_cleaning)
            return curr
        
        self.train = text_clean_wrapper(self.train)
        self.test = text_clean_wrapper(self.test)
        
    def tfidf_vectorizer(self) :
        tfidf_word = TfidfVectorizer(strip_accents = 'unicode', stop_words = 'english', ngram_range = (1,3), max_features = 33000, token_pattern = r'\w{1,}',
                                    min_df = 3, max_df = 0.9, analyzer = 'word')
        
        tfidf_char = TfidfVectorizer(ngram_range = (1,4), min_df = 5, max_df = 0.9, strip_accents = 'unicode', use_idf = True, smooth_idf = True,
                                    sublinear_tf = True, max_features = 25000, analyzer = 'char')

        all_text = self.train['question_text']
        all_text = all_text.append(self.test['question_text'])

        
        tfidf_word.fit(all_text)
        tfidf_char.fit(all_text)
        
        train_word = pd.DataFrame.sparse.from_spmatrix(tfidf_word.transform(self.train['question_text']))
        test_word = pd.DataFrame.sparse.from_spmatrix(tfidf_word.transform(self.test['question_text']))
        
        train_char = pd.DataFrame.sparse.from_spmatrix(tfidf_char.transform(self.train['question_text']))
        test_char = pd.DataFrame.sparse.from_spmatrix(tfidf_char.transform(self.test['question_text'])) 
        
        self.train = pd.concat([train_word, train_char], axis = 1, ignore_index = True, sort = False)
        self.test = pd.concat([test_word, test_char], axis = 1, ignore_index = True, sort = False)
        
    def vt_fit(self) :
        self.vt.fit(self.train, self.y)
        
    def predict(self) :
        self.final_y = self.vt.predict_proba(self.test, weights = [1.1, 3, 0.084, 1.09])
        self.final_y = (self.final_y > 0.27).astype(np.int)
        self.final_y = pd.DataFrame(self.final_y)
        self.final_y['target'] = self.final_y[0]
        self.final_y.drop([0], axis = 1, inplace = True)
        self.final_y['qid'] = self.qid
        
    def get_submission(self) :
        self.final_y.to_csv('submission.csv', index = False)
        
    def to_call(self) :
        self.data_cleaning()
        self.tfidf_vectorizer()
        self.vt_fit()
        self.predict()
        self.get_submission()
        

if __name__ == "__main__" :
    data_viz = data_visualization()
    data_viz.show()
    del(data_viz)
    
    model = pre_processing()
    model.to_call()
    del(model)

