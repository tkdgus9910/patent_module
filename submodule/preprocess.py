# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 10:27:33 2022

@author: tmlab

version: v1.0

"""

import pandas as pd
import numpy as np 
import re 

# 출원인 대표명화
    
    # 데이터 전처리
def wisdomain_prep(data) :
    
    dictionary = {    
        
        # id
        '출원번호' : 'id_application',
        '공개번호' : 'id_publication',
        '등록번호' : 'id_registration',
        '번호' : 'id_wisdomain',
          
        # human information
        '출원인' : 'applicants',
        '출원인대표명' : 'applicant_rep',
        '출원인국가' : 'applicant_country', 
        '발명자' : 'inventor',
        '발명자국가' : 'inventor_country',
        
        
        # classification code
        '국제특허분류' : 'IPC',
        '공통특허분류' : 'CPC',
        '미국특허분류' : 'USPC',
        
        
        # date information
        '출원일' : 'date_application', 
        '공개일' : 'date_publication', 
        '등록일' : 'date_registration', 
        
        # text
        '명칭' : 'title',
        # '명칭(원문)' : 'title',
        
        '요약' : 'abstract',
        # '요약(원문)' : 'abstract',
        
        '전체 청구항' : 'claims', 
        '대표 청구항' : 'claims_rep', 
        
        # familiy
        'INPADOC 패밀리' : 'family_INPADOC',
        'INPADOC패밀리국가' : 'family_INPADOC_country',
           
        # status
        '권리 현황' : 'status_right', 
        '최종 상태' : 'status_final', 
        
        # citation
        '자국인용특허' : 'citation_backward_domestic', 
        '외국인용특허' : 'citation_backward_foreign', 
        '자국피인용횟수' : 'citation_forward_domestic_count', 
        # '외국피인용특허' : 'citation_forward_foreign', 
        }
    
    cols = [i for i in data.columns if i in list(dictionary.keys())]
    
    if 'title' not in cols :
        dictionary['명칭(원문)'] = 'title'
        dictionary['요약(원문)'] = 'abstract'
    
    cols = [i for i in data.columns if i in list(dictionary.keys())]
        
    data = data[cols]
    data.columns = [dictionary[i] for i in data.columns] 
    
    # data = data.dropna(subset = ['application_date']).reset_index(drop = 1)
    
    data['year_application'] = data['date_application'].apply(lambda x : int(x.split('.')[0]) if str(x) != 'nan'  else x)
    
    data['TA'] = data['title'] + '. ' + data['abstract']
    data['TAF'] = data.apply(lambda x: x.TA +' ' + x.claims_rep if str(x.claims_rep) != 'nan' else x.TA, axis= 1)
    
    
    data['CPC_sg'] = data['CPC'].apply(lambda x : x.split(', '))
    data['CPC_mg'] = data['CPC_sg'].apply(lambda x : list(set([i.split('/')[0] for i in x])))
    data['CPC_sc'] = data['CPC_mg'].apply(lambda x : list(set([i[0:4] for i in x])))
    data['CPC_mc'] = data['CPC_sc'].apply(lambda x : list(set([i[0:3] for i in x])))
    
    
    data['applicants'] = data['applicants'].apply(lambda x : x.split('|') if str(x) != 'nan' else x)
    data['applicant_country'] = data['applicant_country'].apply(lambda x : x.split('|') if str(x) != 'nan' else x)
    data['inventor'] = data['inventor'].apply(lambda x : x.split('; ') if str(x) != 'nan' else x)
    data['inventor_country'] = data['inventor_country'].apply(lambda x : x.split('|') if str(x) != 'nan' else x)
    
    
    data['family_INPADOC'] = data['family_INPADOC'].apply(lambda x : x.split(', ') if str(x) != 'nan' else x)
    data['family_INPADOC_country'] = data['family_INPADOC_country'].apply(lambda x : x.split(',') if str(x) != 'nan' else x)
    data['family_INPADOC_country_count'] = data['family_INPADOC_country'].apply(lambda x : len(set(x)) if str(x) != 'nan' else x)
    
    
    data['citation_backward_domestic'] = data['citation_backward_domestic'].apply(lambda x : x.split(', ') if str(x) != 'nan' else x)
    data['citation_backward_foreign'] = data['citation_backward_foreign'].apply(lambda x : x.split(', ') if str(x) != 'nan' else x)
    
    #
    applicants_list = data['applicants'].apply(lambda x : [i.lower() for i in x] if str(x) != 'nan' else x)
    applicants_list = prep_applicants(applicants_list)
    data['applicants'] = applicants_list
    
    
    applicant_rep_list = data['applicant_rep'].apply(lambda x : x.lower() if str(x) != 'nan' else x)
    applicant_rep_list = prep_applicant_rep(applicant_rep_list)
    data['applicant_rep'] = applicant_rep_list
    
    data['inventor'] = data['inventor'].apply(lambda x : [i.lower() for i in x] if str(x) != 'nan' else x)
    
    # 패밀리특허 중복 확인
    switch = 0 
    
    for idx, row in data.iterrows() : 
        if str(row['family_INPADOC']) != 'nan' :
            for family in row['family_INPADOC'] :
                
                if family in data['id_publication'] :
                    switch = 1
                    break
                else : pass
                
            if switch == 1 : 
                print('Duplicate family(INPADOC) patents exist.')
                break
            else : pass
            
    if switch == 0 : 
        print('Duplicate family(INPADOC) patents not exist.')
    
    
    data = data.reset_index(drop = 1)
    
    return(data)
    
def prep_applicant_rep(applicant_rep_list) :
    
    stopwords = ['ltd', 'llc', 'inc', 'corp', 'na', 'gmbh', 'co', 
                 'ag', 'usaa', 'lp', 'sa', 'se', 'nv', 'plc']    
    
    for stop in stopwords : 
        pattern = r"\b{}\b".format(stop)    
        applicant_rep_list = [re.sub(pattern, "", i) if str(i) != 'nan' else i  for i  in applicant_rep_list  ]        
    
    applicant_rep_list = [i.split('|')[0] if str(i) != 'nan' else i  for i  in applicant_rep_list  ]        
    applicant_rep_list = [re.sub("  ", " ", i).strip() if str(i) != 'nan' else i  for i  in applicant_rep_list  ]        
        
    return(applicant_rep_list)

def prep_applicants(applicants_list) :
        
    stopwords = ['ltd', 'llc', 'inc', 'corp', 'na', 'gmbh', 'co', 
                 'ag', 'usaa', 'lp', 'sa', 'se', 'nv', 'plc']    
    
    for stop in stopwords : 
        pattern = r"\b{}\b".format(stop)    
        applicants_list = [[re.sub(pattern, "", i) for i in x ] if str(x) != 'nan' else x for x  in applicants_list  ]        
        
    applicants_list = [[re.sub("  ", " ", i).strip() for i in x ] if str(x) != 'nan' else x for x  in applicants_list  ]        
        
    return(applicants_list)


    

    
