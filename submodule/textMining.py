# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 12:06:36 2023

@author: tmlab
"""

from collections import Counter
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
import nltk
from gensim import models
from gensim.corpora.dictionary import Dictionary
from gensim.test.utils import common_corpus, common_dictionary
import gensim
from gensim import corpora
from collections import Counter


stopwords = stopwords.words('english')

# AO extract
def get_function(doc) :
    
    # INPUT : nlp(doc)
    # OUTPUT : list of fucntion
    
    function_list = []
    
    for token in doc:
            # print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
                    # token.shape_, token.is_alpha, token.is_stop)        
        if token.pos_ == 'VERB' : 
            # print(token.text, token.dep_, token.head.text, token.head.pos_,[(child.text, child.pos_, child.dep_) for child in token.children])
            
            for child in token.children : 
                # [(child.text, child.pos_, child.dep_) for child in token.children]
                if 'obj' in child.dep_ :
                    function = token.lemma_ + '_' +child.lemma_ 
                    function_list.append(function)
                    
    return(function_list)

# ADV+V extraction
def get_condition(doc) : 
    
    # INPUT : nlp(doc)
    # OUTPUT : list of condition
        
    condition_list = []
    
    for token in doc:
            # print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
                    # token.shape_, token.is_alpha, token.is_stop)        
        if token.pos_ == 'VERB' : 
            print(token.text, token.dep_, token.head.text, token.head.pos_,[(child.text, child.pos_, child.dep_) for child in token.children])
            
            for child in token.children : 
                # [(child.text, child.pos_, child.dep_) for child in token.children]
                if 'ADV' in child.pos_ :
                    function = token.lemma_ + '_' + child.lemma_
                    condition_list.append(function)
                    
    return(condition_list)

def sw_filtering_bigram(function_list) :
    
    # INPUT : list of bigram_words
    # OUTPUT : list of bigram_words
    
    # stopwords filtering
    stopwords = ['claim', 'accord', 'comprise', 'include','have', 'plurality', 'say', 'far', 'configure', 'wherein', 'that', 'one', 'some',
                 'current', 'either' ,'then','thereby','yet', 'once','so', 'also','only','thereof','first','therein','prior','currently']
    
    function_list_ = []
    
    for func in function_list : 
        func = func.split("_")
        
        if (func[0] not in stopwords) and (func[1] not in stopwords) : 
            function_list_.append("_".join(func))
            
    return(function_list_)


# TF-IDF
def tfidf_counter(word_list, tf_min = 3, df_max = -1) :
    
    # INPUT : LIST TYPE OF WORD
    # OUTPUT : TF-IDF
    temp = [item for sublist in word_list for item in sublist]
    c = Counter(temp)
    
    counter = pd.DataFrame(c.items())
    counter = counter.sort_values(1, ascending=False).reset_index(drop = 1)
    counter.columns = ['term', 'tf']
    counter = counter[counter['tf'] >= tf_min].reset_index(drop = 1)
    counter['df'] = 0
    
    for idx,row in counter.iterrows() :
        term = row['term']
        for temp_list in word_list :
            if term in temp_list : counter['df'][idx] +=1
    
    if df_max != -1 : 
        counter = counter[counter['df'] <= df_max].reset_index(drop = 1)
        
    # for idx,row in counter.iterrows() :
    #     if row['df'] >= 1 :
            
    
    counter['tf-idf'] = counter['tf'] / np.sqrt((1+ counter['df']))
    counter = counter.sort_values('tf-idf', ascending=False).reset_index(drop = 1)
    #counter = counter.loc[counter['tf-idf'] >= 1.5 , :].reset_index(drop = 1)
    return(counter)


# 약어 처리 함수
def get_abbrev_df(texts):
    
    abbrev_df = pd.DataFrame()
    
    for text in texts : 
        text = text.split(" ")
        for idx, word in enumerate(text) :
            
            pat = r'\(+[A-Z]+?\)+' # (word) 패턴        
            temp = re.match(pattern= pat, string = word)
            
            if type(temp) == re.Match :
                # print(1)
                
                abbrev = temp.string
                pat = r"[^A-Z]+"
                abbrev = re.sub(pattern= pat, repl= '' ,string = abbrev).strip()
                
                if len(abbrev) < 2 : continue
                
                original = ''
                
                for i in range(len(abbrev)) :
                    idx_ = idx-(len(abbrev)-i)
                    keyword = re.sub(pattern = '-' , repl = ' ', string = text[idx_])
                    original += keyword
                    if len(original.split(" ")) >= len(abbrev) : break
                
                    original += ' '
                
                original = original.lower().strip()
                test = original.split()
                test = [i[0] for i in test]
                test = "".join(test)
                if test == abbrev.lower() :
                
                    abbrev_df = abbrev_df.append({'abbrev' : abbrev,
                                                  'original' : original}, ignore_index = 1)
                # abbrev_dict[abbrev] = original
                
    return(abbrev_df)
    
    
    
# 약어df를 기반으로 사전 생성
def get_abbrev_dict(texts, minimum_cnt) :
    
    abbrev_df = get_abbrev_df(texts)

    c = Counter(abbrev_df.abbrev)
    c = {x: count for x, count in c.items() if count >= minimum_cnt}
    abbrev_list = list(c.keys())
    abbrev_dict = {}
    
    for abbrev in abbrev_list : 
      original_list = abbrev_df.loc[abbrev_df['abbrev'] == abbrev,'original'].tolist()
      c = Counter(original_list)
      original = c.most_common(1)[0][0]
      abbrev_dict[abbrev] = original
    
    
    return(abbrev_dict)


# 약어 dict를 기반으로 약어 치환
def abbrev2origin(abbrev_dict, texts) : 
    
    pat = r'\(+.+?\)+' # () 패턴 제거
    texts = [re.sub(pat, ' ', i) for i in texts]
    
    pat = r"\s\s+" # 더블 스페이스 제거
    texts = [re.sub(pat, ' ', i) for i in texts]
    texts = [i.strip() for i in texts]
    
    for k,v in abbrev_dict.items() :
        pat = k
        repl = v
        texts = [re.sub(pattern = pat, repl = repl, string = i) for i in texts]
        
    return(texts)

# 특수문자 제거
def removing_sc(texts) :
    
    pat = r"[^a-zA-Z0-9 ]" 
    texts = [re.sub(pat, ' ', i) for i in texts]
    
    pat = r"\s\s+" # 더블 스페이스 제거
    texts = [re.sub(pat, ' ', i) for i in texts]
    
    texts = [i.strip() for i in texts]
    
    return(texts)
    

# LDA
def get_word_list(nlp_list) :
    
    nltk.download('stopwords')
    
    # removing stopwords
    word_lists = [[token.lemma_.lower() for token in i] for i in nlp_list]

    # c = Counter([x for xs in word_lists for x in set(xs)])
    
    word_lists_ = [[word for word in inner_list if word.casefold() not in stopwords] for inner_list in word_lists]
    
    stopwords_ = ['comrpise', 'method', 'include', 'one', 'wherein', 
                  'system', 'base', 'first', 'device', 'associate',
                  'least', 'plurality', 'may', 'whether', 'example',
                  'set', 'within', 'indicate', 'disclose', 'relate',
                  'result', 'also', 'say', 'herein', 'among']
    
    word_lists_ = [[word for word in inner_list if word.casefold() not in stopwords_] for inner_list in word_lists_]
    
    return(word_lists_)
    
class LDA_gensim :
    
    def __init__(self, word_lists) :
        
        self.word_list = word_lists
        self.dictionary = Dictionary(word_lists)
        self.dictionary.filter_extremes(no_below = 5) # 5번이하 등장은 제거
        self.corpus = [common_dictionary.doc2bow(text) for text in word_lists]
        self.passes = 1
        self.alpha = 'auto'
        self.eta = 'auto'
        self.k = 20
        self.model_list = []
        
    def tunning_passes(self) :
        
        corpus = self.corpus
        dictionary = self.dictionary
        k = self.k
        alpha = self.alpha
        eta = self.eta
        self.model_list = []
        
        # 1. passes를 결정 
        # Define the hyperparameters
        chunksize = 2000
        iterations = 400
        eval_every = None
        
        # Tune the hyperparameters using the grid search
        
        perplexity_rank = []
        
        for num_topics in range(10, 51, 10):
            
            perplexity_scores = []
            
            for passes in range(1, 11, 1):
                lda_model = gensim.models.ldamodel.LdaModel(
                    corpus= corpus,
                    id2word=dictionary,
                    num_topics=num_topics,
                    random_state=100,
                    update_every=1,
                    chunksize=chunksize,
                    passes=passes,
                    iterations=iterations,
                    alpha=alpha,
                    eta=eta,
                    per_word_topics=True)
                self.model_list.append((num_topics, passes, lda_model))
                perplexity_score = lda_model.log_perplexity(corpus)
                perplexity_scores.append(perplexity_score)
                print(passes, num_topics)
            
            min_value = min(perplexity_scores)
            min_index = perplexity_scores.index(min_value)+1 #best index
            perplexity_rank.append(min_index)
        
        counter = Counter(perplexity_rank)
        most_common_value = counter.most_common(1)[0][0]
        
        print("best passes is {}".format(self.passes))
        
        self.passes = most_common_value
        
    def tunning_ab(self) :
        
        # 2. alpha, beta를 결정
        
        corpus = self.corpus
        dictionary = self.dictionary
        k = self.k
        self.model_list = []
        
        # Define the hyperparameters
        chunksize = 2000
        iterations = 400
        eval_every = None
        passes = self.passes
        
        # Tune the hyperparameters using the grid search
        perplexity_rank = []
        
        
        alphas = list(np.logspace(-3, 0, 10))
        alphas = [round(i, 3) for i in alphas]
        # alphas.append('auto')
        betas = list(np.logspace(-3, 0, 10))
        betas = [round(i, 3) for i in betas]
        # betas.append('auto')
        
        perplexity_scores = []
        
        for alpha in alphas :
            for beta in betas :
                lda_model = gensim.models.ldamodel.LdaModel(
                    corpus=corpus,
                    id2word=dictionary,
                    num_topics = k,
                    random_state=100,
                    update_every=1,
                    chunksize=chunksize,
                    passes = passes,
                    iterations=iterations,
                    alpha=alpha,
                    eta=beta,
                    per_word_topics=True)
                self.model_list.append((alpha, beta, lda_model))
                perplexity_score = lda_model.log_perplexity(corpus)
                perplexity_scores.append((alpha, beta, perplexity_score))
                print(alpha, beta)
        
        scores = [i[2] for i in perplexity_scores]
        min_value = min(scores)
        min_index = scores.index(min_value) #best index
        
        self.alpha = perplexity_scores[min_index][0]
        self.eta = perplexity_scores[min_index][1]
        
        print("best alpha,eta is {},{}".format(self.alpha, self.eta))
        
        
    def tunning_k(self) :
        
        corpus = self.corpus
        dictionary = self.dictionary
        alpha = self.alpha
        eta = self.eta
        passes = self.passes
        self.model_list = []
        
        chunksize = 2000
        iterations = 400
        eval_every = None
        
        # 3. topic k를 결정
        perplexity_scores = []
        
        for k in range(4, 101, 4):
            lda_model = gensim.models.ldamodel.LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics = k,
                random_state=100,
                update_every=1,
                chunksize=chunksize,
                passes= passes,
                iterations=iterations,
                alpha=alpha,
                eta=eta,
                per_word_topics=True)
            
            self.model_list.append((k, lda_model))
            perplexity_score = lda_model.log_perplexity(corpus)
            print("iteration : {}, pereplexity : {}".format(k,perplexity_score))
            
            perplexity_scores.append((k, perplexity_score))
            
    
        scores = [i[1] for i in perplexity_scores]
        min_value = min(scores)
        min_index = scores.index(min_value) #best index
        
        self.k = perplexity_scores[min_index][0]
        print("best k is {}".format(self.k))
                
        
        
    def get_topic_doc(lda_model, corpus) :
        
        topic_doc_df = pd.DataFrame(columns = range(0, lda_model.num_topics))
        
        for corp in corpus :
            
            temp = lda_model.get_document_topics(corp)
            DICT = {}
            for tup in temp :
                DICT[tup[0]] = tup[1]
            
            topic_doc_df = topic_doc_df.append(DICT, ignore_index=1)
        topic_doc_df = np.array(topic_doc_df)
        topic_doc_df = np.nan_to_num(topic_doc_df)
        
        
        return(topic_doc_df)
    
    def get_topic_word_matrix(lda_model) :
        
        topic_word_df = pd.DataFrame()
        
        for i in range(0, lda_model.num_topics) :
            temp = lda_model.show_topic(i, 1000)
            DICT = {}
            for tup in temp :
                DICT[tup[0]] = tup[1]
                
            topic_word_df = topic_word_df.append(DICT, ignore_index =1)
            
        topic_word_df = topic_word_df.transpose()
        
        return(topic_word_df)
    
    def get_topic_topword_matrix(lda_model, num_word) :
        
        topic_word_df = pd.DataFrame()
        
        for i in range(0, lda_model.num_topics) :
            temp = lda_model.show_topic(i, num_word)
            temp = [i[0] for i in temp]
            DICT = dict(enumerate(temp))
            
            topic_word_df = topic_word_df.append(DICT, ignore_index =1)
            
        topic_word_df = topic_word_df.transpose()
        
        return(topic_word_df)
        
        
        
    
    
    