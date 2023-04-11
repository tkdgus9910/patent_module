# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 12:06:36 2023

@author: tmlab
"""

from collections import Counter
import pandas as pd
import numpy as np
import re

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
    