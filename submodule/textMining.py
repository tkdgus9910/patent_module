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
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
import gensim
from sklearn.preprocessing import RobustScaler
from scipy.stats import entropy

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
    

# 4LDA
def get_word_list(nlp_list) :
    
    nltk.download('stopwords')
    
    # removing stopwords
    word_lists = [[token.lemma_.lower() for token in i] for i in nlp_list]
    word_lists = [[token for token in i if len(token) >= 3] for i in word_lists]

    # c = Counter([x for xs in word_lists for x in set(xs)])
    
    word_lists_ = [[word for word in inner_list if word.casefold() not in stopwords] for inner_list in word_lists]
    
    stopwords_ = ['comrpise', 'method', 'include', 'one', 'wherein', 
                  'system', 'base', 'first', 'device', 'associate',
                  'least', 'plurality', 'may', 'whether', 'example',
                  'set', 'within', 'indicate', 'disclose', 'relate',
                  'result', 'also', 'say', 'herein', 'among', 
                  'another', 'current', 'second', 'first', 'third', 'part',
                  'side', 'portion', 'member', 'unit']
    
    word_lists_ = [[word for word in inner_list if word.casefold() not in stopwords_] for inner_list in word_lists_]
    
    return(word_lists_)
    


    
def normalize2one(values):
    total = sum(values)
    normalized = [value / total for value in values]
    return normalized



class LDA_gensim :
    
    def __init__(self, word_lists) :
        
        self.word_list = word_lists
        self.dictionary = Dictionary(word_lists)
        self.dictionary.filter_extremes(no_below = 3) # 5번이하 등장은 제거
        self.corpus = [self.dictionary.doc2bow(text) for text in word_lists]
        self.passes = 2
        self.alpha = 0.1
        self.eta = 0.1
        self.k = 20
        self.random_state = 1234
        
        self.model_list = []
        
        self.model_final = gensim.models.ldamodel.LdaModel(
                    corpus= self.corpus,
                    id2word = self.dictionary,
                    num_topics= self.k,
                    random_state = self.random_state,
                    passes = self.passes,
                    alpha = self.alpha,
                    eta = self.eta,
                    per_word_topics = True,
                    )
        
    def tunning_passes(self, method = ['diversity', 'coherence']) :
        
        self.model_list = []
        
        # 1. passes를 결정 
        
        # Tune the hyperparameters using the grid search
        scores_df = pd.DataFrame()
        
        for passes in range(2, 11, 2):
            
            lda_model = gensim.models.ldamodel.LdaModel(
                corpus= self.corpus,
                id2word = self.dictionary,
                num_topics= self.k,
                random_state = self.random_state,
                passes = passes,
                alpha = self.alpha,
                eta = self.eta,
                per_word_topics = True,)
            
            self.model_final = lda_model
            self.model_list.append((passes, lda_model))
            score_dict = {}
            score_dict['iter'] = passes
            score_dict.update(self.evaluate_model(lda_model, method)) 
            
            scores_df = scores_df.append(score_dict, ignore_index = 1)
            print("passes {}".format(passes))
        
        # scaler = RobustScaler()
        # scaled_data = scaler.fit_transform(scores_df.iloc[:,1:])
        # scaled_df = pd.DataFrame(scaled_data)
        scaled_df = scores_df.iloc[:,1:]
        scaled_df = scaled_df / scaled_df.max()
        
        scores_df['score_final'] = np.mean(scaled_df, axis = 1)
        
        # result_df = pd.concat([scores_df, scaled_df], axis = 1)
        max_index = scores_df['score_final'].idxmax()
        
        self.passes = int(scores_df['iter'][max_index])
        
        print("best passes is {}".format(self.passes))
        
        self.refresh_model()
        
        return(scores_df)
        
        
    def tunning_ab(self, method = ['diversity', 'coherence']) :
        
        # 2. alpha, beta를 결정
        
        scores_df = pd.DataFrame()
        self.model_list = []
        
        # Tune the hyperparameters using the grid search
        alphas = list(np.logspace(-3, 0, 7))
        alphas = [round(i, 3) for i in alphas]
        
        betas = list(np.logspace(-3, 0, 7))
        betas = [round(i, 3) for i in betas]
        
        
        for alpha in alphas :
            for eta in betas :
                
                lda_model = gensim.models.ldamodel.LdaModel(
                    corpus= self.corpus,
                    id2word = self.dictionary,
                    num_topics= self.k,
                    random_state = self.random_state,
                    passes = self.passes,
                    alpha = alpha,
                    eta = eta,
                    per_word_topics = True,
                    )
                
                self.model_final = lda_model
                self.model_list.append((alpha, eta, lda_model))
                score_dict = {}
                score_dict['iter'] = [alpha, eta]
                score_dict.update(self.evaluate_model(lda_model, method)) 
                scores_df = scores_df.append(score_dict, ignore_index = 1)
                
                print("alpha, eta {},{}".format(alpha, eta))
            
        # scaler = RobustScaler()
        # scaled_data = scaler.fit_transform(scores_df.iloc[:,1:])
        # scaled_df = pd.DataFrame(scaled_data)
        scaled_df = scores_df.iloc[:,1:]
        scaled_df = scaled_df / scaled_df.max()
        
        scores_df['score_final'] = np.mean(scaled_df, axis = 1)
        
        # result_df = pd.concat([scores_df, scaled_df], axis = 1)
        max_index = scores_df['score_final'].idxmax()
            
        self.alpha = scores_df['iter'][max_index][0]
        self.eta = scores_df['iter'][max_index][1]
                
        print("best alpha, eta = {},{}".format(self.alpha, self.eta))
        
        self.refresh_model()
        
        return(scores_df)
        
        
        
    def tunning_k(self, method = ['diversity', 'coherence']) :
        
        self.model_list = []
        
        # 3. topic k를 결정
        scores_df = pd.DataFrame()
        
        unit = 5
        k = unit
        score_dict_before = {}
        stack = 0
        
        while 1 :
            
            lda_model = gensim.models.ldamodel.LdaModel(
                corpus= self.corpus,
                id2word = self.dictionary,
                num_topics= k,
                random_state = self.random_state,
                passes = self.passes,
                alpha = self.alpha,
                eta = self.eta,
                per_word_topics = True,
                )
            
            self.model_final = lda_model
            self.model_list.append((k, lda_model))
            
            score_dict = {}
            score_dict['iter'] = k
            score_dict.update(self.evaluate_model(lda_model, method)) 
            
            scores_df = scores_df.append(score_dict, ignore_index = 1)
            print("k = {}".format(k))
            
            # 수렴여부 판단
            
            if k != unit :
                score_dict_now = score_dict
                growth_rate_list = []
                for key in score_dict.keys() :
                    if key != 'iter' :
                        score_now = score_dict_now[key]
                        score_before = score_dict_before[key]
                        growth_rate_list.append((abs(score_now)-abs(score_before)) / abs(score_before))
                        
                
                print(growth_rate_list)
                
                if all(growth_rate < 0.025 for growth_rate in growth_rate_list) : 
                    stack +=1
                    
                else :
                    stack = 0
                
                print(stack)
                
                if stack >= 3 : break
                    
            k += unit
            score_dict_before = score_dict
            
        
        # scaler = RobustScaler()
        # scaled_data = scaler.fit_transform(scores_df.iloc[:,1:])
        # scaled_df = pd.DataFrame(scaled_data)
        scaled_df = scores_df.iloc[:,1:]
        scaled_df = scaled_df / scaled_df.max()
        
        scores_df['score_final'] = np.mean(scaled_df, axis = 1)
        
        max_index = scores_df['score_final'].idxmax()
            
        self.k = int(scores_df['iter'][max_index])
        print("best k is {}".format(self.k))
        
        self.refresh_model()
        
        return(scores_df)
        
                
    def get_docByTopics(self) :
        
        corpus = self.corpus
        model = self.model_final
        docByTopics = pd.DataFrame(columns = range(0, model.num_topics))
        
        for corp in corpus :
            temp = model.get_document_topics(corp, minimum_probability = 0)
            DICT = {}
            for tup in temp :
                DICT[tup[0]] = tup[1]
            docByTopics = docByTopics.append(DICT, ignore_index=1)
            
        docByTopics = np.array(docByTopics)
        self.docByTopics = np.nan_to_num(docByTopics)
        
        return(self.docByTopics)
    
    def get_topicProportion(self) : 
        
        # docTopic_matrix = self.get_docTopic_matrix()
        topic_size = self.docByTopics.sum(axis =0)
        topic_prop = normalize2one(topic_size)
        self.topicProportion = topic_prop
        
        return(self.topicProportion)  
        
    def get_wordByTopics(self) :
        
        model = self.model_final
        
        topic_word_df = pd.DataFrame()
        
        for i in range(0, model.num_topics) :
            temp = model.show_topic(i, 1000)
            DICT = {}
            for tup in temp :
                DICT[tup[0]] = tup[1]
                
            topic_word_df = topic_word_df.append(DICT, ignore_index =1)
            
        topic_word_df = topic_word_df.transpose()
        
        return(topic_word_df)
    
    def get_topwordByTopics(self, num_word = 30) :
        
        model = self.model_final
        
        topic_word_df = pd.DataFrame()
        
        for i in range(0, model.num_topics) :
            temp = model.show_topic(i, num_word)
            temp = [i[0] for i in temp]
            DICT = dict(enumerate(temp))
            
            topic_word_df = topic_word_df.append(DICT, ignore_index =1)
            
        topic_word_df = topic_word_df.transpose()
        return(topic_word_df)
    
    def get_topwordthetaByTopics(self, num_word = 30) :
        
        model = self.model_final
        
        topic_word_df = pd.DataFrame()
        
        for i in range(0, model.num_topics) :
            temp = model.show_topic(i, num_word)
            temp = [i[1] for i in temp]
            DICT = dict(enumerate(temp))
            
            topic_word_df = topic_word_df.append(DICT, ignore_index =1)
            
        topic_word_df = topic_word_df.transpose()
        return(topic_word_df)
    
    def CAGR(first, last, periods): 
        first = first+1
        last = last+1
        return (last/first)**(1/periods)-1  
    
    def get_topic_cutoff(self, method) :
        
        temp =  np.ravel(self.docByTopics, order='C').tolist()
        
        if method == 'iqr' : 
            # cut-off 계산    
            Q1 = np.quantile(temp, 0.25)
            Q3 = np.quantile(temp, 0.75)
            IQR = np.quantile(temp, 0.75) - np.quantile(temp, 0.25)
            
            cut_off = Q3 + 1.5*IQR #2.4 #1.5 # 1.17 #0.8
        
        elif method == 'sigma' :
            cut_off = np.mean(temp)+ 2*np.std(temp) # mu + sigma
        
        self.topic_cutoff = cut_off
        
        return (self.topic_cutoff)
    
    def get_topic_time(data, col_time, col_topics) : 
        
        result_df = pd.DataFrame()
        
        for platform in set(data[col_time]) :
            
            temp_data_ = data.loc[data[col_time] == platform,:].reset_index(drop = 1)
            _list = temp_data_[col_topics].tolist()
            
            c = Counter(c for clist in _list for c in clist)
            # c = sorted(c.items(),key = lambda i: i[0])
            result_df = result_df.append(c, ignore_index =  1)
            
        result_df.index = set(data[col_time])
        result_df = result_df.sort_index(axis=1)
        result_df = result_df.fillna(0)
        
        return(result_df)
    
    def get_summary_topic_time(topic_time_df, recent_period) :
        
        end_year = max(topic_time_df.index)
        start_year= min(topic_time_df.index)
        total_period = end_year - start_year +1
        recent_year = end_year-(recent_period-1)
        
        # print(recent_year)
        
        topic_time_df_recent = topic_time_df.loc[topic_time_df.index >= recent_year,:]
        
        volume = topic_time_df.apply(lambda x : np.sum(x))
        volume_recent = topic_time_df_recent.apply(lambda x : np.sum(x))
        
        cagr = topic_time_df.apply(lambda x : CAGR(x[start_year], x[end_year], total_period))
        cagr_recent = topic_time_df_recent.apply(lambda x : CAGR(x[recent_year], x[end_year], recent_period))
        
        result = pd.concat([volume, volume_recent, cagr, cagr_recent], axis = 1)
        
        result.columns = ['volume', 'volume_recent', 'CAGR', 'CAGR_recent']
        result['topic'] = result.index      
        
        result = result[['topic','volume', 'volume_recent', 'CAGR', 'CAGR_recent']]
        
        return(result)
    
    def get_most_similar_doc2topic(self, data_sample, 
                                   top_n = 5, 
                                   title = 'title', 
                                   date = 'date') :
        
        result_df = pd.DataFrame()
        title = title
        
        for col in range(self.docByTopics.shape[1]) :
            
            DICT = {}
        
            for n in range(1, top_n+1) :
                
                idx = self.docByTopics.argsort(axis = 0)[-n][col]
                DICT['topic'] = col
                DICT['rank'] = n
                DICT['title'] = data_sample[title][idx]
                DICT['date'] = data_sample[date][idx]
                DICT['similarity'] = self.docByTopics[idx,col]
            
                result_df = result_df.append(DICT, ignore_index=1)
                
        self.most_similar_doct2topic = result_df
        
        return(result_df)
    
        
    def refresh_model(self) : 
        
        self.model_final = gensim.models.ldamodel.LdaModel(
                    corpus= self.corpus,
                    id2word = self.dictionary,
                    num_topics= self.k,
                    random_state = self.random_state,
                    passes = self.passes,
                    alpha = self.alpha,
                    eta = self.eta,
                    per_word_topics = True,
                    )
        
        
    def evaluate_model(self, model, measure_list) :
        
        model = model
        corpus = self.corpus
        dictionary = self.dictionary
        word_list = self.word_list
        result = {}
        
        if 'perplexity' in measure_list :
            score = abs(model.log_perplexity(corpus))
            result['perplexity'] = score
            
        # 토픽 내 단어 분포의 일관성
        if 'coherence' in measure_list :
            
            coherence_model = CoherenceModel(model=model, 
                                             texts=word_list, 
                                             dictionary=dictionary, 
                                             coherence='u_mass', topn=10)
            
            score = abs(coherence_model.get_coherence())
            result['coherence'] = score
            
        if 'diversity' in measure_list :
            
            # keyword_set = set()
            # for topic in range(model.num_topics):
            #     keyword_list = [i[0] for i in model.show_topic(topic, 10)]
            #     for keyword in keyword_list : 
            #         keyword_set.add(keyword)
                
            # score = len(keyword_set) / (model.num_topics*10)
            
            
            docTopic_matrix = self.get_docByTopics()
            topic_size = docTopic_matrix.sum(axis =0)
            topic_prop = normalize2one(topic_size)
            topwordTopics_matrix = self.get_topwordByTopics(10)
            topwordthetaTopics_matrix = self.get_topwordthetaByTopics(10)
            
            for col in topwordthetaTopics_matrix.columns : 
                topwordthetaTopics_matrix[col] = normalize2one(topwordthetaTopics_matrix[col]) 
                 
            result_dict = {}
            
            for col in topwordTopics_matrix.columns : 
                
                weight = topic_prop[col]
                
                for idx, word in enumerate(topwordTopics_matrix[col]) : 
                # word = topwordTopics_matrix[col]
                    value = topwordthetaTopics_matrix[col][idx] * weight
                    if word in result_dict.keys() : 
                        result_dict[word].append(value)
                    else : 
                        result_dict[word] = [value]
                        
            for key in result_dict.keys() :
                result_dict[key] = np.sum(result_dict[key])
            
            base = 2  # work in units of bits
            temp = list(result_dict.values())
            score = entropy(temp, base=base)
            
            result['diversity'] = score
            
        
        # 토픽 간 평균 유사도
        return(result)
    
    def save(self, directory, file_name, data) :
        
        
        writer = pd.ExcelWriter(directory + 'LDA_results_'+file_name+'.xlsx', 
                                engine = 'xlsxwriter')
        
        data.to_excel(writer , sheet_name = 'data', index = 1)
        pd.DataFrame(self.docTopic_matrix).to_excel(writer , sheet_name = 'docByTopics', index = 1)
        pd.DataFrame(self.topic_prop.items(), columns = ['topic', 'volumn']).to_excel(writer , sheet_name = 'topic_proportion', index = 1)
        
        self.topwordTopic_matrix .to_excel(writer , sheet_name = 'topwordByTopics', index = 1)
        self.title.to_excel(writer , sheet_name = 'titleByTopics', index = 1)
        self.wordTopic_matrix.to_excel(writer , sheet_name = 'wordByTopics', index = 1)
        # result.to_excel(writer , sheet_name = 'topic_volume_cagr', index = 1)
        
        writer.save()
        writer.close()
        
        
    
        
            
        
def CAGR(first, last, periods): 
    first = first+1
    last = last+1
    return (last/first)**(1/periods)-1  
    
    
    