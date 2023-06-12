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
    file_name = 'sample_2'
    data = pd.read_csv(directory+ file_name+ '.csv', skiprows = 4)
    
    data['file_name'] = file_name
    
    # 데이터 전처리    
    from preprocess import wisdomain_prep
    
    data_ = wisdomain_prep(data)    
    
    #%% 1-2. !optional : scraping description
    
    import scraper
    
    data_['description'] = dict
    
    for idx, row in data_.iloc[:,:].iterrows() : 
        
        print(idx)
        pt_id = row['id_registration']
        data_['description'][idx] = scraper.scraping_description(pt_id)
    
    data_['description_'] = scraper.preprocessing_header(data_['description'], 5)

    #%% 특허지표 계산
    
    import indicator  
    
    # 등록특허로 필터링
    data_registered = data_.loc[data_['id_registration'] != np.nan , :]
        
    # 주체별
    cpp = indicator.calculate_CPP(data_registered, 'applicant_rep', 'citation_forward_domestic_count')
    pii = indicator.calculate_PII(data_registered, 'applicant_rep', 'citation_forward_domestic_count')
    ts = indicator.calculate_TS(data_registered, 'applicant_rep', 'citation_forward_domestic_count')
    pfs = indicator.calculate_PFS(data_registered, 'applicant_rep', 'family_INPADOC_country_count')
    
    data_applicants = pd.concat([cpp,pii,ts,pfs], axis=1)
    data_applicants.columns = ['cpp','pii','ts','pfs']
    
    # 분야전체
    cr4 = indicator.calculate_CRn(data_registered, 
                                  'applicant_rep', 
                                  'citation_forward_domestic_count', 
                                  4)
        
    hhi = indicator.calculate_HHI(data_registered, 
                                  'applicant_rep', 
                                  'id_publication')
    
    cagr = indicator.calculate_CAGR(data_registered, 
                                  'year_application', 
                                  2000,
                                  2020)
    
    #%% 4. 텍스트 분석 준비
    
    import spacy
    import textMining
    import time 
    
    start_time = time.time()
    
    # p1) removing abbreviation(optional)
    abbrev_dict = textMining.get_abbrev_dict(data_['TAF'], 3)
    data_['TAF'] = textMining.abbrev2origin(abbrev_dict , data_['TAF'])
    
    # p2) removing speical character(optional)
    data_['TAF'] = textMining.removing_sc(data_['TAF'])
    
    # p3) change type 2 nlp
    nlp = spacy.load("en_core_web_sm")
    
    data_['TAF_nlp'] = data_['TAF'].apply(lambda x : nlp(x))

    end_time = time.time()
    execution_time = end_time - start_time
    
    print("코드 실행 시간: ", execution_time, "초")
    
    #%% 4-1 LDA analysis
    
    import textMining
    
    # LDA 적합
    word_lists = textMining.get_word_list(data_['TAF_nlp'])
    LDA_0 = textMining.LDA_gensim(word_lists)
    
    tunning_df = LDA_0.tunning_passes(['perplexity','diversity', 'coherence']) 
    tunning_df = LDA_0.tunning_ab(['perplexity','diversity', 'coherence'])
    #%%
    tunning_df = LDA_0.tunning_k(method = ['perplexity','diversity', 'coherence'])
    
    #%%
    
    
    
    #%%
    
    # 결과확인    
    # LDA_0.alpha = 0.01
    # LDA_0.refresh_model()
    
    docTopic_matrix = LDA_0.get_docByTopics()
    wordTopic_matrix = LDA_0.get_wordByTopics()
    topwordTopic_matrix = LDA_0.get_topwordByTopics()
    topic_prop = LDA_0.get_topicProportion()

    
    title = LDA_0.get_most_similar_doc2topic(data, title = 'title', date = 'Year')
    
    
    #%%
    LDA_0.k = 200
    LDA_0.refresh_model()
    
    
    #%%
    import pyLDAvis.gensim_models
    import pyLDAvis
    
    # 결과 변경
    LDA_0.alpha = 0.01
    LDA_0.refresh_model()

    # LDAVis 
    vis_data = pyLDAvis.gensim_models.prepare(LDA_0.model_final, 
                                              LDA_0.corpus,
                                              LDA_0.dictionary
                                              )
    
    pyLDAvis.save_html(vis_data, directory+ 'test.html')
    
    
    
    
    #%% 4-2 SAO analysis
    
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
    
    
    
    
    
    #%% 5. 연관규칙_네트워크 분석
    
    import ARM
    import pandas as pd
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import apriori, association_rules
    
    item_list = data_['function_list'].tolist()
    frequent_itemsets = ARM.list2apriori(item_list)
    edge_df = ARM.filtering_apriori(frequent_itemsets, 'lift', 1.5)
        
    
    #%% 6. Claims_representative LDA
    
    import re 
    # p1) removing abbreviation(optional)
    abbrev_dict = textMining.get_abbrev_dict(data_['claims_rep'], 2)
    data_['claims_rep'] = textMining.abbrev2origin(abbrev_dict , data_['claims_rep'])
    #%%
    
    data_['claims_rep_list'] = data_['claims_rep'].apply(lambda x : re.split(';|:',x)[1:])
    data_['claims_rep_list'] = data_['claims_rep_list'].apply(lambda x : [i.strip() for i in x])
    data_['claims_rep_list'] = data_['claims_rep_list'].apply(lambda x : [i for i in x if len(i)>= 10])
    
    
    #%%
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP
    from hdbscan import HDBSCAN
    from transformers.pipelines import pipeline
    
    # Define custom preprocessor function to remove numeric words
    def remove_numeric_words(text):
        return re.sub(r'\b\d+\b', '', text)

    #%%
    
    embedding_model = pipeline("feature-extraction",
                               model="AI-Growth-Lab/PatentSBERTa")
    
    umap_model = UMAP(n_neighbors=15, 
                      n_components=50, 
                      min_dist=0.0, 
                      metric='cosine', 
                      random_state=1234)
        
    hdbscan_model = HDBSCAN(min_cluster_size=20, 
                            metric='euclidean', 
                            cluster_selection_method='eom', 
                            prediction_data=True, 
                            min_samples=5)

    my_additional_stop_words = ['invention', 'patent', 'method', 'apparatus', 'apparatuses',
                                'process','application', 'claim', 'priority', 'enablement', 'art', 'background',
                                'comprising', 'consisting', 'wherein', 'embodiment', 'present','preferred',
                                'device', 'base', 'inventive', 'aspect', 'current', 'parts', 'part',
                                'characteristic', 'example', 'for', 'disclosure', 'examples', 'this', 'patent',]
        
    cv = CountVectorizer(stop_words="english")
    default_stopwords = set(cv.get_stop_words())
    stop_words = my_additional_stop_words +list(default_stopwords)
    
    vectorizer_model = CountVectorizer(stop_words=stop_words,
                                       preprocessor=remove_numeric_words)
    
    topic_model = BERTopic(embedding_model=embedding_model,
                           calculate_probabilities=False,
                           umap_model = umap_model,
                           hdbscan_model=hdbscan_model,
                           vectorizer_model=vectorizer_model,
                           top_n_words = 30,
                           min_topic_size = 10)
    
    corpus = data_['claims_rep_list'].tolist()
    corpus = sum(corpus, [])
    
    topics, probs = topic_model.fit_transform(corpus)
    
    #%%
    temp = topic_model.get_topic_info()
    
    #%% YAKE를 이용한 용어 처리 - 테스트중
    
    from yake import KeywordExtractor
    
    language = 'en'
    max_ngram_size = 3
    deduplication_threshold = 0.7
    window_size = 1
    top_k = 20
    
    # YAKE 객체 생성
    kw_extractor = KeywordExtractor(lan=language,
                                        n=max_ngram_size,
                                        dedupLim=deduplication_threshold,
                                        windowsSize=window_size,
                                        top=top_k)
    
    
    # 텍스트 문서
    text = data_['TAF'][0]
    
    # 키워드 추출
    keywords = kw_extractor.extract_keywords(text)
    
    # 추출된 키워드 확인
    for keyword in keywords:
        print(keyword[0])  # 키워드 텍스트
        print(keyword[1])  # 키워드 점수
    
    
    result = []
    
    for doc in data_['claims_rep'] :
        # doc = data_['claims_rep'][0]
        
        keywords = kw_extractor.extract_keywords(doc)
        result.append(keywords)

    
    result_dict = dict()
    
    for k_list in result :
        for k_tuple in k_list :
            if k_tuple[0] not in result_dict.keys() :
                result_dict[k_tuple[0]] = [k_tuple[1]]
            else : 
                result_dict[k_tuple[0]].append(k_tuple[1])

    result_df = pd.DataFrame()
    
    for k in result_dict.keys() : 
        if len(result_dict[k]) < 5 : continue 
        else : result_df = result_df.append({'keyword' : k,
                                             'score_list' : result_dict[k]}, ignore_index = 1)
    
    result_df['score_mean'] = result_df['score_list'].apply(lambda x : np.mean(x))
    result_df['score_count'] = result_df['score_list'].apply(lambda x : len(x))
    