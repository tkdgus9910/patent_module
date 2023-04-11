# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 15:07:38 2023

@author: tmlab
"""

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


def list2apriori(item_list, min_support = 0.005) :
    dataset = item_list

    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    frequent_itemsets = apriori(df, min_support= min_support, use_colnames=True,max_len = 2) # 변경
    
    return(frequent_itemsets)


def filtering_apriori(frequent_itemsets,
                      metric = 'lift' ,
                      threshold = 1.5) : 
    
    result = association_rules(frequent_itemsets, metric= metric, min_threshold= threshold) # 변경
    
    edge_df = result[['antecedents', 'consequents', 'lift']]
    
    edge_df['antecedents'] = edge_df['antecedents'].apply(lambda x : str(x).split("'")[1])
    edge_df['consequents'] = edge_df['consequents'].apply(lambda x : str(x).split("'")[1])
    
    return(edge_df)