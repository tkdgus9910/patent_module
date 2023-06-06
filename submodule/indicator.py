# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 10:27:33 2022

@author: tmlab

version: v1.0

"""

import pandas as pd
import numpy as np 

def calculate_CPP(data, group, citation_forward_count) :
    # CPP, Cites per patent
    return(data.groupby(group)[citation_forward_count].mean())

def calculate_PII(data, group, citation_forward_count) :
    # PII, Patent impact index
    cpp = calculate_CPP(data, group, citation_forward_count)
    mean = data[citation_forward_count].mean()
    return(cpp/mean)

def calculate_TS(data, group, citation_forward_count) :
    # TS, Technology strength
    count = data.groupby(group)[citation_forward_count].count()
    pii = calculate_PII(data, group, citation_forward_count)
    return(pii*count)

def calculate_PFS(data, group, family_country_count) :
    # PFS, Patent familiy size
    mean = data[family_country_count].mean()    
    return(data.groupby(group)[family_country_count].mean()/mean)

def calculate_CRn(data, group, citation_forward_count,n) :
    # CRn, Concentrate Ratio
    count = data.groupby(group)[citation_forward_count].count()
    total = data[citation_forward_count].count().sum()
    share = sorted(count/total, reverse = 1) 
    return(sum(share[0:n]))

def calculate_HHI(data, group, id_publication) : 
    count = data.groupby(group)[id_publication].count()
    total = data.shape[0]
    share = [i/total*100 for i in count]
    hhi = sum([i*i for i in share])
    
    return(hhi)

def CAGR(first, last, periods): 
    first = first+1
    last = last+1
    return (last/first)**(1/periods)-1 

def calculate_CAGR(data, year_application, min_year, max_year) : 
    
    # min_year = min(data[year_application])
    # max_year = max(data[year_application])
    
    periods = max_year-min_year
    first = data.loc[data[year_application] == min_year , :].shape[0]
    last = data.loc[data[year_application] == max_year , :].shape[0]
    
    return(CAGR(first, last, periods))