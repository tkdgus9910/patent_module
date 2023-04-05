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
    import re
    import pandas as pd
    import numpy as np     
    import pickle
    from datetime import datetime
    from datetime import timezone
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
    
    
    #%% 1. 데이터 전처리    
    
    from preprocess import wisdomain_prep
    
    # from wisdomain import wisdomain_prep
    data_ = wisdomain_prep(data)    
    

    #%% 2. 주체별 특허지표 계산
    import indicator
    
    # 등록특허로 필터링
    data_registered = data_.loc[data_['id_registration'] != np.nan , :]
        
    cpp = indicator.calculate_CPP(data_registered, 'applicant_rep', 'citation_forward_domestic_count')
    pii = indicator.calculate_PII(data_registered, 'applicant_rep', 'citation_forward_domestic_count')
    ts = indicator.calculate_TS(data_registered, 'applicant_rep', 'citation_forward_domestic_count')
    pfs = indicator.calculate_PFS(data_registered, 'applicant_rep', 'family_INPADOC_country_count')
    
    
    #%% 3. 분야 전체의 특성 계산
    
    
    cr4 = indicator.calculate_CRn(data_registered, 
                                  'applicant_rep', 
                                  'citation_forward_domestic_count', 
                                  4)
    
    
        
    hhi = indicator.calculate_HHI(data_registered, 
                                  'applicant_rep', 
                                  'id_publication')
    
    
    
    #%% 4.  출원인 대표명화 - test
    import requests
    import xmltodict
    from requests.utils import quote
    
    applicant_rep = quote('INTEL CORP') # percent encoding
    
    url = 'https://assignment-api.uspto.gov/patent/lookup?query=' + applicant_rep
    url += '&filter=OwnerName'
    res = requests.get(url, verify=False)
    
    
    result = xmltodict.parse(res.content)
    