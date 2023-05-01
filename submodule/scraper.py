# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 12:06:36 2023

@author: tmlab
"""

import requests
from bs4 import BeautifulSoup
from bs4 import element
import time
from collections import OrderedDict
import re
from collections import Counter

def scraping_description(id_registration) :
    
    pt_id = id_registration
    url = 'https://patents.google.com/patent/'+ pt_id +'/en'
    soup = BeautifulSoup(requests.get(url).content, 'html.parser')
    time.sleep(1)
    result = OrderedDict()
    
    navigate = soup.find('heading')

    if type(navigate) != element.Tag : pass

    else : 
    
        header = navigate.text
        
        result[header] = []
        
        while(1) : 
            
            if navigate.name == 'section' : break
            if type(navigate) != element.Tag : break        
            
            if ((navigate.name == 'heading') & (navigate.text.isupper() == 1)) : 
                
                header = navigate.text.strip()
                result[header] = []
                    
            elif (navigate.name == 'div')  : 
                result[header].append(navigate.text.strip())
                
            navigate = navigate.find_next()
            
    return(result)

def preprocessing_header(descriptions, top_n = 5) : 
    
    header_list = [list(i.keys()) if i != dict else [] for i in descriptions ]
    
    pat = r'-'
    header_list = [[re.sub(pat, " ", i) for i in l ] for l in header_list ]
    
    pat = r'OF[A-Z ]+'
    header_list = [[re.sub(pat, "", i) for i in l ] for l in header_list ]
    
    pat = r'TO[A-Z ]+'
    header_list = [[re.sub(pat, "", i) for i in l ] for l in header_list ]
    
    pat = r'\bART\b'
    header_list = [[re.sub(pat, "", i) for i in l ] for l in header_list ]
    
    pat = r'\bTECHNICAL FIELD\b'
    header_list = [[re.sub(pat, "FIELD", i) for i in l ] for l in header_list ]
    
    pat = r'\bBRIEF SUMMARY\b'
    header_list = [[re.sub(pat, "SUMMARY", i) for i in l ] for l in header_list ]
    
    header_list = [[i.strip() for i in l ] for l in header_list ]
    
    c = Counter([x for xs in header_list for x in set(xs)])
    c = c.most_common(top_n)
    c = [i[0] for i in c]
    
    header_list = [[i if i in c else 'ETC' for i in l] for l in header_list ]

    for idx, dt in enumerate(descriptions) :
        if dt != dict :
            original_keys = list(dt.keys())
            for idx_, key in enumerate(original_keys) :
                new_key = header_list[idx][idx_]
                # old_key = 
                descriptions[idx][new_key] = descriptions[idx].pop(key)
    
    return(descriptions)
