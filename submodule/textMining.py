# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 12:06:36 2023

@author: tmlab
"""

# AO extract
def get_function(doc) :
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

def sw_filtering(function_list) :
    
    # stopwords filtering
    stopwords = ['claim', 'accord', 'comprise', 'include','have', 'plurality', 'say', 'far', 'configure', 'wherein', 'that', 'one', 'some',
                 'current', 'either' ,'then','thereby','yet', 'once','so', 'also','only','thereof','first','therein','prior','currently']
    
    function_list_ = []
    
    for func in function_list : 
        func = func.split("_")
        
        if (func[0] not in stopwords) and (func[1] not in stopwords) : 
            function_list_.append("_".join(func))
            
    return(function_list_)



    
    
    #%%
    