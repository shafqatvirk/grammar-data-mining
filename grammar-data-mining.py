
import sys,os
import tempScripts2 as parser
from sklearn.externals import joblib
from stanfordcorenlp import StanfordCoreNLP
from joblib import load
import nltk,re

CORE_NLP_DIR = './stanford-corenlp-full-2018-01-31/' 
PARSER = StanfordCoreNLP(CORE_NLP_DIR, memory='8g', lang='en')
print("Loading prelearned encoding...latest")
vec =load("./models/Encoding_fe_identifier-train-tab-CP3.joblib")
vec_classifier = load("./models//Encoding_fe_classifier-train-tab-CP3.joblib")
print("Loading a pretrained cross validated Logistic regression model")    
LReg = LReg = load("./models/LogReg_fe_identifier-train-tab-CP3.joblib")
LReg_classifier = load("./models/LogReg_fe_classifier-train-tab-CP3.joblib")
print('models loaded')

props = {'annotators': 'tokenize,pos,lemma,depparse', 'tokenize.whitespace': 'True', 'ssplit.isOneSentence': 'True'}
PARSER.annotate('dummy sentence', properties=props) # to load the parser in advance

def build_frames_dict():
    frames = open('lus.txt').readlines()
    # frames = open('data/lus.txt').readlines()
    return {l.split('\t')[0]: l.split('\t')[2].rstrip() for l in frames}
frames_dict = build_frames_dict()



def annotate_doc(doc,feature):
    sents = nltk.sent_tokenize(doc)
    if feature == "38A Indefinite Articles":
        return feature_38a(sents)
    elif feature == "37A Definite Articles":
        return feature_37a(sents)
    
    doc_annotations = []
    for sent in sents:
        #print(sent)
        try:
            annotations = parser.server(PARSER, sent,frames_dict,vec,LReg,vec_classifier,LReg_classifier)
            #print(annotations)
            if annotations != [] and annotations != None :
                for (f,l,fes) in annotations:
                    if f == 'SEQUENCE':    
                        if feature == "87A Order of Adjective and Noun":
                            order = feature_87a(fes)
                        elif feature == "81A Order of Subject, Object and Verb":
                            order = feature_81a(fes)
                        elif feature == "86A Order of Genitive and Noun":
                            order = feature_86a(fes)
                        elif feature == "89A Order of Numeral and Noun":
                            order = feature_89a(fes)
                        elif feature == "90A Order of Relative Clause and Noun":
                            order = feature_90a(fes)
                        elif feature == "82A Order of Subject and Verb":
                            order = feature_82a(fes)
                        
            if order != None:
                #print(file_name)
                return order
        except:
            #print('exception')
            continue
        
def feature_87a(fes):
    N_entity_1,A_entity_1,N_entity_2,A_entity_2 = False, False, False, False
    precede,follow = False, False
    NA, AN = False, False
    order = 'Empty'
    frequency = []
    for (fen,fess) in fes:
                                #print(fen,fess)
                                fess = ' '.join([ff.lower() for ff in fess])
                                #print(fess)
                                #fess = ' '.join(fess)
                                if fen == 'Entity_1' and 'noun' in fess:
                                    N_entity_1 = True
                                    #print(N_entity_1)
                                if fen == 'Entity_1' and 'adjective' in fess:
                                    A_entity_1 = True
                                    
                                
                                if fen == 'Entity_2' and 'noun' in fess:
                                    N_entity_2 = True
                                if fen == 'Entity_2' and 'adjective' in fess:
                                    A_entity_2 = True
                                
                                    
                                if fen == 'Order':
                                    if 'precede' in fess:
                                        precede = True
                                    if 'follow' in fess:
                                        follow = True
                                if fen == 'Frequency':
                                    frequency = fess
                                    
    if N_entity_1 and A_entity_2:
                            if precede:
                                if 'normally' in frequency or 'usually' in frequency or 'often' in frequency or 'sometimes' in frequency or 'mostly' in frequency:
                                    order = 'Both'
                                else:
                                    order = 'NA'
                            elif follow:
                                if 'normally' in frequency or 'usually' in frequency or 'often' in frequency or 'sometimes' in frequency or 'mostly' in frequency:
                                    order = 'Both'
                                else:
                                    order = 'AN'
    elif N_entity_2 and A_entity_1:
                            if precede:
                                if 'normally' in frequency or 'usually' in frequency or 'often' in frequency or 'sometimes' in frequency or 'mostly' in frequency:
                                    order = 'Both'
                                else:
                                    order = 'AN'
                            elif follow:
                                if 'normally' in frequency or 'usually' in frequency or 'often' in frequency or 'sometimes' in frequency or 'mostly' in frequency:
                                    order = 'Both'
                                else:
                                    order = 'NA'
                        #print(sent)
    return order
                        #if order != 'Empty':
                         #   print(feature)
                          #  print(order)
def feature_81a(fes):
    for (fen,fess) in fes:
        fess = ' '.join([ff.lower() for ff in fess])
        if fen == 'Order' and 'object' in fess and 'subject' in fess and 'verb' in fess:
            return fess
        
def feature_86a(fes):
    G_entity_1,N_entity_1,G_entity_2,N_entity_2 = False, False, False, False
    precede,follow = False, False
    GN, NG = False, False
    order = 'Empty'
    frequency = []
    for (fen,fess) in fes:
                                #print(fen,fess)
                                fess = ' '.join([ff.lower() for ff in fess])
                                #print(fess)
                                #fess = ' '.join(fess)
                                if fen == 'Entity_1' and 'genetive' in fess:
                                    G_entity_1 = True
                                    #print(N_entity_1)
                                if fen == 'Entity_1' and 'noun' in fess:
                                    N_entity_1 = True
                                    #print(A_entity_1)
                                
                                
                                if fen == 'Entity_2' and 'noun' in fess:
                                    N_entity_2 = True
                                if fen == 'Entity_2' and 'genetive' in fess:
                                    G_entity_2 = True
                                
                                    
                                if fen == 'Order':
                                    if 'precede' in fess:
                                        precede = True
                                    if 'follow' in fess:
                                        follow = True
                                if fen == 'Frequency':
                                    frequency = fess
                                    
    if G_entity_1 and N_entity_2:
                            if precede:
                                if 'normally' in frequency or 'usually' in frequency or 'often' in frequency or 'sometimes' in frequency or 'mostly' in frequency:
                                    order = 'Both'
                                else:
                                    order = 'GN'
                            elif follow:
                                if 'normally' in frequency or 'usually' in frequency or 'often' in frequency or 'sometimes' in frequency or 'mostly' in frequency:
                                    order = 'Both'
                                else:
                                    order = 'NG'
    elif G_entity_2 and N_entity_1:
                            if precede:
                                if 'normally' in frequency or 'usually' in frequency or 'often' in frequency or 'sometimes' in frequency or 'mostly' in frequency:
                                    order = 'Both'
                                else:
                                    order = 'NG'
                            elif follow:
                                if 'normally' in frequency or 'usually' in frequency or 'often' in frequency or 'sometimes' in frequency or 'mostly' in frequency:
                                    order = 'Both'
                                else:
                                    order = 'GN'
                        #print(sent)
    return order       
def feature_89a(fes):
    N_entity_1,Num_entity_1,N_entity_2,Num_entity_2 = False, False, False, False
    precede,follow = False, False
    NumN, NNum = False, False
    order = 'Empty'
    frequency = []
    for (fen,fess) in fes:
                                #print(fen,fess)
                                fess = ' '.join([ff.lower() for ff in fess])
                                #print(fess)
                                #fess = ' '.join(fess)
                                if fen == 'Entity_1' and 'noun' in fess:
                                    N_entity_1 = True
                                    #print(N_entity_1)
                                if fen == 'Entity_1' and 'numeral' in fess:
                                    Num_entity_1 = True
                                    
                                if fen == 'Entity_2' and 'noun' in fess:
                                    N_entity_2 = True
                                if fen == 'Entity_2' and 'numeral' in fess:
                                    Num_entity_2 = True
                                
                                    
                                if fen == 'Order':
                                    if 'precede' in fess:
                                        precede = True
                                    if 'follow' in fess:
                                        follow = True
                                if fen == 'Frequency':
                                    frequency = fess
                                    
    if N_entity_1 and Num_entity_2:
                            if precede:
                                if 'normally' in frequency or 'usually' in frequency or 'often' in frequency or 'sometimes' in frequency or 'mostly' in frequency:
                                    order = 'Both'
                                else:
                                    order = 'NNum'
                            elif follow:
                                if 'normally' in frequency or 'usually' in frequency or 'often' in frequency or 'sometimes' in frequency or 'mostly' in frequency:
                                    order = 'Both'
                                else:
                                    order = 'NumN'
    elif N_entity_2 and Num_entity_1:
                            if precede:
                                if 'normally' in frequency or 'usually' in frequency or 'often' in frequency or 'sometimes' in frequency or 'mostly' in frequency:
                                    order = 'Both'
                                else:
                                    order = 'NumN'
                            elif follow:
                                if 'normally' in frequency or 'usually' in frequency or 'often' in frequency or 'sometimes' in frequency or 'mostly' in frequency:
                                    order = 'Both'
                                else:
                                    order = 'NNum'
                        #print(sent)
    return order

def feature_90a(fes):
    N_entity_1,RC_entity_1,N_entity_2,RC_entity_2 = False, False, False, False
    precede,follow = False, False
    RN, NR = False, False
    order = 'Empty'
    frequency = []
    for (fen,fess) in fes:
                                #print(fen,fess)
                                fess = ' '.join([ff.lower() for ff in fess])
                                #print(fess)
                                #fess = ' '.join(fess)
                                if fen == 'Entity_1' and 'noun' in fess:
                                    N_entity_1 = True
                                    #print(N_entity_1)
                                if fen == 'Entity_1' and 'relative clause' in fess:
                                    RC_entity_1 = True
                                    
                                if fen == 'Entity_2' and 'noun' in fess:
                                    N_entity_2 = True
                                if fen == 'Entity_2' and 'relative clause' in fess:
                                    RC_entity_2 = True
                                
                                    
                                if fen == 'Order':
                                    if 'precede' in fess:
                                        precede = True
                                    if 'follow' in fess:
                                        follow = True
                                if fen == 'Frequency':
                                    frequency = fess
                                    
    if N_entity_1 and RC_entity_2:
                            if precede:
                                if 'normally' in frequency or 'usually' in frequency or 'often' in frequency or 'sometimes' in frequency or 'mostly' in frequency:
                                    order = 'Both'
                                else:
                                    order = 'NR'
                            elif follow:
                                if 'normally' in frequency or 'usually' in frequency or 'often' in frequency or 'sometimes' in frequency or 'mostly' in frequency:
                                    order = 'Both'
                                else:
                                    order = 'RN'
    elif RC_entity_1 and N_entity_2:
                            if precede:
                                if 'normally' in frequency or 'usually' in frequency or 'often' in frequency or 'sometimes' in frequency or 'mostly' in frequency:
                                    order = 'Both'
                                else:
                                    order = 'RN'
                            elif follow:
                                if 'normally' in frequency or 'usually' in frequency or 'often' in frequency or 'sometimes' in frequency or 'mostly' in frequency:
                                    order = 'Both'
                                else:
                                    order = 'NR'
                        #print(sent)
    return order

def feature_82a(fes):
    S_entity_1,V_entity_1,S_entity_2,V_entity_2 = False, False, False, False
    precede,follow = False, False
    SV, VS = False, False
    order = 'Empty'
    frequency = []
    for (fen,fess) in fes:
                                #print(fen,fess)
                                fess = ' '.join([ff.lower() for ff in fess])
                                #print(fess)
                                #fess = ' '.join(fess)
                                if fen == 'Entity_1' and 'subject' in fess:
                                    S_entity_1 = True
                                    #print(N_entity_1)
                                if fen == 'Entity_1' and 'verb' in fess:
                                    V_entity_1 = True
                                    
                                if fen == 'Entity_2' and 'subject' in fess:
                                    S_entity_2 = True
                                if fen == 'Entity_2' and 'verb' in fess:
                                    V_entity_2 = True
                                
                                    
                                if fen == 'Order':
                                    if 'precede' in fess:
                                        precede = True
                                    if 'follow' in fess:
                                        follow = True
                                if fen == 'Frequency':
                                    frequency = fess
                                    
    if S_entity_1 and V_entity_2:
                            if precede:
                                if 'normally' in frequency or 'usually' in frequency or 'often' in frequency or 'sometimes' in frequency or 'mostly' in frequency:
                                    order = 'Both'
                                else:
                                    order = 'SV'
                            elif follow:
                                if 'normally' in frequency or 'usually' in frequency or 'often' in frequency or 'sometimes' in frequency or 'mostly' in frequency:
                                    order = 'Both'
                                else:
                                    order = 'VS'
    elif V_entity_1 and S_entity_2:
                            if precede:
                                if 'normally' in frequency or 'usually' in frequency or 'often' in frequency or 'sometimes' in frequency or 'mostly' in frequency:
                                    order = 'Both'
                                else:
                                    order = 'VS'
                            elif follow:
                                if 'normally' in frequency or 'usually' in frequency or 'often' in frequency or 'sometimes' in frequency or 'mostly' in frequency:
                                    order = 'Both'
                                else:
                                    order = 'SV'
                        #print(sent)
    return order

def feature_37a(sents):
    for sent in sents:
        res1 = re.findall(r"(\w+) is used as definite article",sent)
        if res1 != []:
            return res1[0]
        res2 = re.findall(r"(\w+) definite article",sent)
        if res2 != []:
            return res2[0]
    return None
        
    
def feature_38a(sents):
    for sent in sents:
        res1 = re.findall(r"(\w+) is used as indefinite article",sent)
        if res1 != []:
            return res1[0]
        res2 = re.findall(r"(\w+) indefinite article",sent)
        if res2 != []:
            return res2[0]
    return None


if __name__ == '__main__':
    doc = open('./txt/'+sys.argv[3]+'.txt',errors='ignore').read()   
    annotations = annotate_doc(doc,sys.argv[2])    
    

