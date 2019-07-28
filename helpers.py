from subprocess import STDOUT,PIPE
import os.path,subprocess
import json, re
from stanfordcorenlp import StanfordCoreNLP
from nltk.tree import ParentedTree
from nltk import Tree


#CORE_NLP_DIR = '/Users/virk/Downloads/stanford-corenlp-full-2018-01-31/'    
#PARSER = StanfordCoreNLP(CORE_NLP_DIR, memory='8g', lang='en')
props={'annotators': 'lemma'}
print('healpers loaded')
def compute_head2(newtree):
    tree_nodes_file =  open('node-tree.txt', 'w')
    tree_nodes_list = []
    for subtree in newtree.subtrees():
        tree_nodes_file.write(str(subtree))
        tree_nodes_list.append(str(subtree))
    tree_nodes_file.close()
    java_file = 'Test'
    cmd = 'java -cp .:./stanford-parser-full-2014-01-04/stanford-parser.jar:./headFinder/ ' + java_file 
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    
    return dict(zip(tree_nodes_list,str(out.decode('utf-8')).split('\n')))
    #print(out)
def find_target_attribs(newtree,word,PARSER):
    for subtree in newtree.subtrees():
                                if subtree.label() in ['CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN',
                                               'NNS','NNP',	'NNPS','PDT','POS','PRP','PRP$','RB','RBR','RBS',
                                               'RP'	,'SYM','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT',
                                               'WP','WP$','WRB']: 
            
                                    w,pos= subtree.pos()[0]
                                    #print(w,word)
                                    if w == word:
                                        target_tree = subtree
                                        target_lemma = json.loads(PARSER.annotate(w,properties=props))["sentences"][0]['tokens'][0]['lemma']
                                        target_pos = pos
                                        return(target_tree,target_lemma,target_pos)
    return (None,None,None)

def find_subcat(target_tree):
    return str(target_tree.parent().productions()[0]).replace(' ','')
def find_left_right_word_attribs(subtree,head_tree):
    if len(subtree.pos()) == 1:
                                    left_word,left_word_pos = 'NA','NA'
                                    right_word,right_word_pos = 'NA','NA'
    elif head_tree.pos()[0] == subtree.pos()[0]:
                                    left_word,left_word_pos = subtree.pos()[-1]
                                    right_word,right_word_pos = 'NA','NA'
    elif head_tree.pos()[0] == subtree.pos()[-1]:
                                    right_word,right_word_pos = 'NA','NA'
                                    left_word,left_word_pos = subtree.pos()[0]
    else:
                                    right_word,right_word_pos = subtree.pos()[-1]
                                    left_word,left_word_pos = subtree.pos()[0]
    return (left_word,left_word_pos,right_word,right_word_pos)

def compute_position(subtree,target_tree):
    for (a,b) in (zip(subtree.treeposition(),target_tree.treeposition())):
        if a == b:
            continue
        elif a < b:
            return 'L'
        else:
            return 'R'
    return 'O'

def compute_gov_cat(subtree):
    #print(subtree)
    if subtree.label() in ['S','VP','SINV','SQ','ROOT']:
        return subtree.label()
    else:
        return compute_gov_cat(subtree.parent())
def compute_parent_attribs(subtree,parent,heads_dict):
    if parent == None:
        parent_word,parent_word_pos='ROOT','ROOT'
    else:
                                    
        parent_head = heads_dict[str(subtree.parent())]
                                    
        parent_head_tree = ParentedTree.convert(Tree.fromstring(parent_head.rstrip()))
        parent_word,parent_word_pos=parent_head_tree.pos()[0]
    return (parent_word,parent_word_pos)

def path_finder(subtree,target_node):
    path = VisitNode(subtree,target_node)
    if path != None:
            #print(subtree)
            return '-'.join(path)
            #print('#'*50)
    else:
            temp_path = []
            
            while (path==None):
                if subtree.parent() == None:
                    break
                temp_path = temp_path + [subtree.label()] 
                subtree = subtree.parent()
                
                path = VisitNode(subtree,target_node)
                if path != None :
                    path = temp_path + path
                    
                    break
            return '-'.join(path)
                
        
    
def VisitNode(node, target):
    #print(node)
    # Base case. If we found the target, return target in a list
    if node == target:
        return [node.label()]

    # If we're at a leaf and it isn't the target, return None 
    if node.height() == 2:
        return None

    # recursively iterate over children
    #children = node.subtrees()
    #print(children)
    for i in node:
        #print(i)
        tail = VisitNode(i, target)
        
        if tail: # is not None
            return [node.label()] + tail # prepend node to path back from target
    return None #none of the children contains target

def build_fes_dict():
    from itertools import groupby
    from operator import itemgetter
    fes_file = open('frame_elements.txt').readlines()
    fes_tuple_list = [tuple(l.rstrip().split('\t')) for l in fes_file]
    #print(fes_tuple_list)
    return {k : '#and#'.join(list(list(zip(*g))[1])) for k, g in groupby(fes_tuple_list, itemgetter(0))}
                          