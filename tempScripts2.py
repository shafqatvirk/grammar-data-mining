from nltk.tokenize import sent_tokenize
from stanfordcorenlp import StanfordCoreNLP
import json, re
import nltk,os
import itertools
from nltk.corpus import wordnet as wn
import xml.etree.ElementTree as ET
from nltk import Tree
from nltk.tree import ParentedTree
import os.path,subprocess
from subprocess import STDOUT,PIPE
import pandas as pd


import sys,os
#sys.path.insert(0, '../../code/')
###!sys.path.insert(0, '/Users/virk/shafqat/postDoc-Swe/project/lingFN/annotation/parser/code/')
import helpers as helpers



def annotate_sentence(PARSER,sentence,frames_dict,vec,LReg,vec_classifier,LReg_classifier):
    
    #CORE_NLP_DIR = '/Users/virk/Downloads/stanford-corenlp-full-2018-01-31/'    
    #PARSER = StanfordCoreNLP(CORE_NLP_DIR, memory='8g', lang='en')
    props={'annotators': 'tokenize,lemma'}
    frame_fes_dict = helpers.build_fes_dict()
    sentence_annotated_frames = []
    #print(frame_fes_dict)
    all_features = []
    triggered_frames = {}
    for d in json.loads(PARSER.annotate(sentence,properties=props))["sentences"][0]['tokens']:
        
        if d['lemma'] in frames_dict:
            
            triggered_frames[(d['lemma'],d['originalText'])] = frames_dict[d['lemma']]
    
            
    #print(triggered_frames)
   
    try:
                    ##parse = json.loads(PARSER.annotate(sent_text,properties=props))
                        parse = PARSER.parse(sentence)
    except:
        print('error parsing')
        return 'Null'
    
                    
                    #print(type(parse))
    t = Tree.fromstring(parse)
    newtree = ParentedTree.convert(t)
    heads_dict = helpers.compute_head2(newtree)
        
    #print(triggered_frames)                
    for (l,word) in triggered_frames:
        
        #annotated_frames = []
        frame_name = '_'.join(triggered_frames[(l,word)].split('_')[1:])
                        
        target_tree = None
        target_lemma = None
        target_pos = None
        fes_list = frame_fes_dict[frame_name]

        target_tree,target_lemma,target_pos = helpers.find_target_attribs(newtree,word,PARSER)
        if target_tree == None:
                continue
            
        c_subcat = helpers.find_subcat(target_tree)
        
        frame_elements = {}                   
        for subtree in newtree.subtrees():
                                
                                ##fe_identifier_file = open('./data/fe_identifier-train-tab-CP.csv','w+')
                                ##fe_identifier_file.write(','.join(['target_lemma','target_pos','arg_word','arg_word_pos','right_word','right_word_pos','left_word','left_word_pos','parent_word','parent_word_pos','c_subcat','phrase_type','position','fes_list','gov_cat'])+'\n')
                                feature_labels = ['target_lemma','target_pos','arg_word','arg_word_pos','right_word','right_word_pos','left_word','left_word_pos','parent_word','parent_word_pos','c_subcat','phrase_type','position','fes_list','gov_cat','c_path']
    
                                features = []
                                ###open('node-tree.txt', 'w').write(str(subtree))
                                #print(subtree)
                                ###head = compute_head2()
                                if not str(subtree) in heads_dict:
                                    continue
                                head = heads_dict[str(subtree)]
                                if head.rstrip() == ' ' or head.strip() == '':
                                    continue
                                ###head_tree = ParentedTree.convert(Tree.fromstring(head.decode("utf-8").rstrip()))
                                head_tree = ParentedTree.convert(Tree.fromstring(head.rstrip()))
                                #print(subtree.pos())
                                
                                left_word,left_word_pos,right_word,right_word_pos = helpers.find_left_right_word_attribs(subtree,head_tree)
                                arg_word,arg_word_pos = head_tree.pos()[0]
                                parent = subtree.parent()
                                phrase_type = subtree.label()
                                position = helpers.compute_position(subtree,target_tree)
                                gov_cat = helpers.compute_gov_cat(subtree)
                                parent_word,parent_word_pos = helpers.compute_parent_attribs(subtree,parent,heads_dict)
                        
                                
                                path = helpers.path_finder(subtree,target_tree)
                                
                                features = [target_lemma,target_pos,arg_word,arg_word_pos,right_word,right_word_pos,left_word,left_word_pos,parent_word,parent_word_pos,c_subcat,phrase_type,position,fes_list,gov_cat,path]
                                
                                candidate_str = ' '.join(subtree.leaves())
                                
                                #head_tree.pos()
                                ##fe_identifier_file.write(','.join(features)+'\n')
                                ##fe_identifier_file.close()
                                ##X_cols = pd.read_csv("./data/fe_identifier-train-tab-CP.csv")
                               
                                feature_dict = {k:[v] for (k,v) in zip(feature_labels,features)}
                                
                                #print(feature_dict)
                                X_cols =  pd.DataFrame.from_dict(feature_dict)
                                #print(X_cols)
                                X_cols.fillna('NA', inplace=True)
                                X_dict = X_cols.to_dict('records')
    
                                # Now only calling the transform function to turn the data into 
                                # encoded vectors
                                Xpos_vectorized = vec.transform(X_dict)
                                
                                Xpos_vectorized_classifier = vec_classifier.transform(X_dict)
    
                                #print(vec.get_feature_names())
                                X = Xpos_vectorized.toarray()
                                X_classifier = Xpos_vectorized_classifier.toarray()
    
    
                                #%% Load Pretrained LogisticRegression model
                                
                                # Use the model to make predictions.
                                fe_identifier_pred_labels = LReg.predict(X)
    
                                ##print([fe_identifier_pred_labels[0]],subtree.leaves())
    
                                
                                if fe_identifier_pred_labels[0] == 'Y':
                                        fe_identifier_pred_labels_classifier = LReg_classifier.predict(X_classifier)
                                        
                                            
                                        ##print(features)
                                        ##print(frame_name,target_lemma,fe_identifier_pred_labels_classifier[0],subtree.leaves())
                                        #print(frame_fes_dict[frame_name].split('#and#'))
                                        if fe_identifier_pred_labels_classifier[0].lower() in frame_fes_dict[frame_name].split('#and#') or 'fe_'+fe_identifier_pred_labels_classifier[0].lower() in frame_fes_dict[frame_name].split('#and#'):
                                            #print('added')
                                            if fe_identifier_pred_labels_classifier[0] not in ['data','data_translation']:
                                                fe_label = fe_identifier_pred_labels_classifier[0].lstrip('fe_')
                                                if  fe_label in frame_elements.keys():
                                                    #if len(' '.join(subtree.leaves())) > len(frame_elements[fe_label]):
                                                        frame_elements[fe_label] = frame_elements[fe_label] +[' '.join(subtree.leaves())]
                                                else:
                                                    frame_elements[fe_label] = [' '.join(subtree.leaves())]
                                                ##frame_elements.append((fe_identifier_pred_labels_classifier[0].lstrip('fe_'),' '.join(subtree.leaves())))
    
                                #all_features.append(','.join(features))
        frame_elements_l = list(frame_elements.items())
        #print(frame_elements_l)
        if frame_elements_l == []:
                    frame_elements_l = [('{-}','{-}')]
        #sentence_annotated_frames.append((frame_name.upper(),target_lemma,list(set(frame_elements_l))))
        sentence_annotated_frames.append((frame_name.upper(),target_lemma,frame_elements_l))
    
    return sentence_annotated_frames
                                      
            
    #PARSER.close()'''
        
def server(PARSER, sentence,frames_dict,vec,LReg,vec_classifier,LReg_classifier):
#if __name__ == "__main__":
    
    #frames_dict = build_frames_dict()
    # sentence = 'The imperative is however formed from a separate base'
    x = annotate_sentence(PARSER,sentence,frames_dict,vec,LReg,vec_classifier,LReg_classifier)
    return x    
