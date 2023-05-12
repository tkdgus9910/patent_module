# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 14:03:32 2022

version : 1.0

@author: tmlab
"""

# 1. 데이터 로드

if __name__ == '__main__':
    
    import os
    import sys
    import pandas as pd
    import numpy as np     
    import warnings
    
    warnings.filterwarnings("ignore")
    
    directory = os.path.dirname(os.path.abspath(__file__))
    directory = directory.replace("\\", "/") # window
    sys.path.append(directory+'/submodule')
    
    # load data
    directory += '/input/'
    file_name = 'sample'
    data = pd.read_csv(directory+ file_name+ '.csv', skiprows = 4)
    
    data['file_name'] = file_name
    
    # 1. 데이터 전처리    
    
    from preprocess import wisdomain_prep
    
    data_ = wisdomain_prep(data)    
    
    #%% 1-2. option : scraping description
    
    import scraper
    
    data_['description'] = dict
    
    for idx, row in data_.iloc[:,:].iterrows() : 
        
        print(idx)
        pt_id = row['id_registration']
        data_['description'][idx] = scraper.scraping_description(pt_id)
    
    data_['description_'] = scraper.preprocessing_header(data_['description'], 5)

    #%% 2. 주체별 특허지표 계산
    import indicator  
    
    # 등록특허로 필터링
    data_registered = data_.loc[data_['id_registration'] != np.nan , :]
        
    cpp = indicator.calculate_CPP(data_registered, 'applicant_rep', 'citation_forward_domestic_count')
    pii = indicator.calculate_PII(data_registered, 'applicant_rep', 'citation_forward_domestic_count')
    ts = indicator.calculate_TS(data_registered, 'applicant_rep', 'citation_forward_domestic_count')
    pfs = indicator.calculate_PFS(data_registered, 'applicant_rep', 'family_INPADOC_country_count')
    
    data_applicants = pd.concat([cpp,pii,ts,pfs], axis=1)
    data_applicants.columns = ['cpp','pii','ts','pfs']
    
    #%% 3. 분야 전체의 특성 계산
    
    
    cr4 = indicator.calculate_CRn(data_registered, 
                                  'applicant_rep', 
                                  'citation_forward_domestic_count', 
                                  4)
        
    hhi = indicator.calculate_HHI(data_registered, 
                                  'applicant_rep', 
                                  'id_publication')
    
    
    #%% 4. 텍스트 분석 준비
    import spacy
    import textMining
    
    # p1) removing abbreviation(optional)
    abbrev_dict = textMining.get_abbrev_dict(data_['TAF'], 3)
    data_['TAF'] = textMining.abbrev2origin(abbrev_dict , data_['TAF'])
    
    # p2) removing speical character(optional)
    data_['TAF'] = textMining.removing_sc(data_['TAF'])
    
    # p3) change type 2 nlp
    nlp = spacy.load("en_core_web_sm")
    data_['TAF_nlp'] = data_['TAF'].apply(lambda x : nlp(x))
    
    
    
    
    #%% 4-1 SAO analysis
    from collections import Counter 
    
    data_['function_list'] = np.nan #V+O
    
    for idx, row in data_.iterrows() : 
        
        doc = row['TAF_nlp']
        function_list = textMining.get_function(doc)
        function_list = textMining.sw_filtering_bigram(function_list)

        data_['function_list'][idx] = function_list
    
    #c = Counter([x for xs in data_['function_list'] for x in set(xs)])
    
    # get tf-idf AO
    c = textMining.tfidf_counter(data_['function_list'])
    
    #%% 4-2 LDA analysis
    
    import textMining
    
    word_lists = textMining.get_word_list(data_['TAF_nlp'])
    LDA_0 = textMining.LDA_gensim(word_lists)
    LDA_0.tunning_passes()    
    LDA_0.tunning_ab()    
    
    #%%
    
    
    LDA_0.tunning_k()    
    
    #%%
    model_list = LDA_0.model_list
    
    #%%
    model = models.LdaModel(corpus, 
                            id2word=dictionary, 
                            num_topics=30, 
                            alpha = 1, eta = 0.00001, random_state=123456, passes = 10)
    
    # word_lists = [[word for word in x if len(word) >= 3] for x in word_lists]
    # word_lists = [[word for word in x if len(word) <= 20] for x in word_lists]
    # word_lists = [text_preprocessing.removing_sw(x, stopwords) for x in word_lists]

    
    
    topic_doc_df = get_topic_doc(model, common_corpus)
    topic_word_df = get_topic_word_matrix(model)
    topic_topword_df = get_topic_topword_matrix(model, 30) # 상위 단어개수 수정 
    
    #%% 


    #%%    
    
    # 3. topic k를 결정
    model_list = []
    
    for k in range(10, 101, 10):
        lda_model = gensim.models.ldamodel.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics = k,
            random_state=100,
            update_every=1,
            chunksize=chunksize,
            passes= most_common_value,
            iterations=iterations,
            alpha=best_alpha,
            eta=best_beta,
            per_word_topics=True)
        model_list.append((k, lda_model))
        print(k)
            
    #%%
    
    
    perplexity_scores = []
    coherence_scores = []
    
    perplexity_weight = 0.5
    coherence_weight = 0.5
    
    for model in model_list:
        
        perplexity_score = model[1].log_perplexity(corpus)
        perplexity_scores.append((model[0], perplexity_score))
        
        coherence_model_lda = gensim.models.CoherenceModel(
            
            model=model[1],
            texts=word_lists_,
            dictionary=dictionary,
            coherence='c_v')
        
        coherence_score = coherence_model_lda.get_coherence()
        coherence_scores.append((k, coherence_score))
        # weighted_score = perplexity_weight * perplexity_score + coherence_weight * coherence_score
            

        
    #%%
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scores = [i[2] for i in coherence_scores]
    
    
    scores = np.array(scores).reshape(-1,1)
    # scores = scaler.fit_transform(scores)
            
        
            
            
    #%%
    # Evaluate the models using the coherence score
    
        
        #%%
    
    # Find the best model
    best_model = max(scores, key=lambda x: x[2])
    print('Best Model:', best_model)
    
    #%%
    
    
    #%%
    
    coherence_model_lda.get_coherence()
    # model[2].log_perplexity(corpus)
    
    #%% 5. 연관규칙_네트워크 분석
    
    import ARM
    import pandas as pd
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import apriori, association_rules
    
    
    item_list = data_['function_list'].tolist()
    frequent_itemsets = ARM.list2apriori(item_list)
    edge_df = ARM.filtering_apriori(frequent_itemsets, 'lift', 1.5)
        
    
    
    
    #%% 6.  출원인 대표명화 - 작업중
    import requests
    import xmltodict
    from requests.utils import quote
    
    applicant_rep = quote('INTEL CORP') # percent encoding
    
    url = 'https://assignment-api.uspto.gov/patent/lookup?query=' + applicant_rep
    url += '&filter=OwnerName'
    res = requests.get(url, verify=False)
    
    
    result = xmltodict.parse(res.content)
    