#!/usr/bin/env python
import sys
import collections

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context 

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

import numpy as np
import tensorflow

import gensim
import transformers 

import string
from collections import defaultdict 

from typing import List

def tokenize(s): 
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos) -> List[str]:
    # Part 1
    syn_lemmas = set()

    for syn in wn.synsets(lemma, pos):
        for lem in syn.lemmas():
            str = lem.name()
            if '_' in str:
                str = str.replace("_", " ")
            if str != lemma and str not in syn_lemmas:
                syn_lemmas.add(str)

    return list(syn_lemmas)

def smurf_predictor(context : Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context : Context) -> str:
    # Part 2
    syns = wn.synsets(context.lemma, pos = context.pos)
    word_freq = defaultdict(int)

    for syn in syns:
        for lem in syn.lemmas():
            if lem.name().lower() != context.lemma:
                word_freq[lem] = lem.count()

    pred = max(word_freq, key=word_freq.get)

    str = pred.name()

    if '_' in str:
        str = str.replace("_", " ")

    return str

def wn_simple_lesk_predictor(context : Context) -> str:
    # Part 3
    syns = wn.synsets(context.lemma, pos = context.pos)
    cntxt = context.left_context + context.right_context 
    context_full = [x.lower() for x in cntxt]
    score_dict = defaultdict(int)

    for syn in syns:
        word_counts = tokenize(syn.definition())
        for ex in syn.examples():
            word_counts += tokenize(ex) 

        for hyp in syn.hypernyms():
            word_counts += tokenize(hyp.definition())
            for ex in hyp.examples():
                word_counts += tokenize(ex)

        overlap_set = set(word_counts) & set(context_full)

        for x in stopwords.words('english'):
            if x in overlap_set:
                overlap_set.remove(x)

        # calculate overlap score
        overlap = len(overlap_set)
        score_dict[syn] = overlap*1000

        # calculate frequency of target word lexeme in synset
        # calculate frequency of other lexemes in synset
        for lem in syn.lemmas():
            if lem.name() == context.lemma:
                score_dict[syn] += lem.count()*100
            else:
                score_dict[syn] += lem.count()

    str = get_wrd(score_dict,context.lemma)

    if '_' in str:
        str = str.replace("_", " ")

    return str
   
def get_wrd(scores,wrd) -> str:
        
        if len(scores) == 0: #base case, all the synsets have been removed
            return wrd
            
        pred = max(scores,key=scores.get) #find best synset

        lexeme_freq = defaultdict()
        for lem in pred.lemmas():
            if lem.name() == wrd:
                continue
            lexeme_freq[lem] = lem.count()
        
        #all lexemes in synset are target word
        if (len(lexeme_freq) == 0):
            scores.pop(pred)
            str = get_wrd(scores, wrd) #try next best synset
        else:
            str = max(lexeme_freq,key=lexeme_freq.get).name()
        
        return str

class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self,context : Context) -> str:
        candidates = get_candidates(context.lemma, context.pos)
        simil = defaultdict()

        for wrd in candidates:
            # ignores multi-word case
            if wrd not in self.model or wrd == context.lemma:
                continue 
            simil[wrd] = self.model.similarity(context.lemma,wrd)

        pred = max(simil,key=simil.get)

        return pred # replace for part 4
    
    def part6_predict(self, context : Context) -> str:
        # for each candidate, calculate total similarity between each context word and candidate
        # + similarity between target word and candidate
        # ignore stopwords and multi-word case

        candidates = get_candidates(context.lemma, context.pos)
        cand_score = defaultdict(int)
        cntxt = context.left_context + context.right_context
        cntxt_full = [x.lower() for x in cntxt]

        for cand in candidates:
            if cand not in self.model:
                continue 
            cand_score[cand] += self.model.similarity(context.lemma,cand)
            for wrd in cntxt_full:
                if wrd not in self.model or wrd in stopwords.words('english'):
                    continue
                cand_score[cand] += self.model.similarity(cand,wrd)

        pred = max(cand_score,key=cand_score.get)

        return pred


class BertPredictor(object):

    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context : Context) -> str:
        candidates = get_candidates(context.lemma,context.pos)

        sentence = " ".join(context.left_context) + " [MASK] " + " ".join(context.right_context)

        input_toks = self.tokenizer.encode(sentence)
        tokens = self.tokenizer.convert_ids_to_tokens(input_toks)
        input_mat = np.array(input_toks).reshape((1,-1))
        outputs = self.model.predict(input_mat,verbose=0)
        predictions = outputs[0]

        wrd_idx = tokens.index("[MASK]")
        best_words = np.argsort(predictions[0][wrd_idx])[::-1] # sort in increasing order
        best_words = self.tokenizer.convert_ids_to_tokens(best_words)
        pred = best_words[0]

        for wrd in best_words:
            if wrd in candidates:
                pred = wrd
                break

        return pred # replace for part 5
    

if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    #W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    #predictor = Word2VecSubst(W2VMODEL_FILENAME)
    predictor = BertPredictor()

    for context in read_lexsub_xml(sys.argv[1]):
        #print(context)  # useful for debugging

        #prediction = wn_frequency_predictor(context) # part 2
        #prediction = wn_simple_lesk_predictor(context) # part 3
        #prediction = predictor.predict_nearest(context) # part 4
        prediction = predictor.predict(context) # part 5
        #prediction = predictor.part6_predict(context) # part 6

        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
