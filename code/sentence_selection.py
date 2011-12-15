#################################
## Quan Dong (qdong)           ##
## Fabian Peternek (fape)      ##
## Yayang Tian (yaytian)       ##
## CIS 530 - Project - Code    ##
#################################

import os
import math

from nltk.tokenize import sent_tokenize

from ordering import compute_similarity_matrix, cluster_sentences

__fape_collection_base_path = '/home1/c/cis530/data-hw3/articles/'
__fape_files_to_load = 10

__cluto_bin = '../cluto-2.1.2/Darwin-i386/scluster'

#1.2
def load_topic_words(topic_file):
    """ Loads the words from topic file into a dictionary and returns that. """
    f = open(topic_file, 'r')
    # Get word/score combinations into a list
    lines = f.readlines()
    # File is no longer needed
    f.close()
    
    topic_dict = {}
    # Split all the lines, convert to tuples and then put it into the dict
    for (word, score) in map(tuple, map(str.split, lines)):
        topic_dict[word] = float(score)
    return topic_dict

def get_top_n_words(topic_words_dict, n):
    """ Gets the n top scored words from topic_words_dict. """
    score_wordlist = topic_words_dict.items()
    score_wordlist.sort(key=lambda x: x[1], reverse=True)
    return [word for (word,score) in score_wordlist[:n]]

def filter_top_n_words(topic_words_dict, n, word_list):
    """ Filters the top n unique words from the word list, sorted by their
        topic words score. """
    # First remove any redundant words in word_list
    words = set(word_list)
    # Now get the intersection with words, that appear as keys in the dict
    topic_words_intersect = set(topic_words_dict.keys()).intersection(words)
    # Now get the words with their scores, sort descending for the scores
    # and return the first n words:
    score_wordlist = [(x, topic_words_dict[x]) for x in topic_words_intersect]
    score_wordlist.sort(key=lambda x: x[1], reverse=True)
    return [word for (word,score) in score_wordlist[:n]]

#1.4
def load_file_sentences(filepath, filename):
    """ Loads sentences of a file into a list and converts everything to
    lower case. """
    # Read file as string first
    f = open(filepath, 'r')
    text = f.read()
    f.close()
    # Strip the newlines
    text = filter(lambda x: x != '\n', text)
    # Now use nltks method to read the sentences
    sentences = sent_tokenize(text)
    # convert everything to lower case
    sentences = map(str.lower, sentences)
    """sentences = [(s.lower(), filename) for s in sentences]"""
    # Create segments by clustering. Let's say 3 segments per text.
    # Similarity metric shall be cosine.
    fs = create_feature_space(sentences)
    vectors = [vectorize(fs, sent) for sent in sentences]
    compute_similarity_matrix(vectors, cosine_similarity, filename+".similarities")
    segments = cluster_sentences(filename+".similarities", __cluto_bin, 3)
    # Stitch it all together
    return zip(sentences, [filename]*len(sentences), segments)

def load_collection_sentences(collection, n):
    """ collection is a directory containing text-files. The first n of those
        text files will be loaded as sentences. """
    files = os.listdir(collection)
    files_sentences = []
    for f in files:
        files_sentences.append(load_file_sentences(collection + "/" + f,f))
        n -= 1
        if n == 0:
            break
    return files_sentences

def dependency_parse_sentence(sentence):
    """ Returns the dependency parse of the given sentence. """
    p = Parser()
    deps = p.parseToStanfordDependencies(sentence)
    # Flatten and organize in the required format
    return [(r, gov.text, dep.text) for r, gov, dep in deps.dependencies]

def dependency_parse_collection(path):
    """ Parses an entire collection and returns the results """
    collection = load_collection_sentences(path, __fape_files_to_load)
    parse = []
    for text in collection:
        for sentence in text:
            parse.append(dependency_parse_sentence(sentence))
    return reduce(lambda x,y: x+y, parse)

def get_linked_words(dependency_list, word):
    """ Returns a list of unique words which that appear together with word
    in the dependency_list. """
    words = set() # Unique words, so first a set
    for (rel, gov, dep) in dependency_list:
        if gov == word: 
            words.add(dep)
        elif dep == word:
            words.add(gov)
    return list(words)

# 2.1
def create_feature_space(sentences):
    """ create feature space from hw1 solution. Creates a feature space. """
    splits = [s.split() for s in sentences]
    types = set(reduce(lambda x, y: x + y, splits))
    lookup = dict()
    for i, word in enumerate(types):
        lookup[word] = i
    return lookup

def create_collection_feature_space(collection_path):
    """ Creates a feature space for a collection. """
    sentences = load_collection_sentences(collection_path, __fape_files_to_load)
    return create_feature_space(reduce(lambda x,y: x[0]+y[0], sentences))

def vectorize(vector_space, sentence):
    """ vectorize from solution to hw1. Creates a vector for the given sentence
    in accordance to the given feature space. """
    vector = [0] * len(vector_space)
    for word in sentence[0].split():
        vector[vector_space[word]] = 1
    return vector

def vectorize_collection(feature_space, collection_path):
    """ Vectorizes an entire collection. """
    sentences = load_collection_sentences(collection_path, __fape_files_to_load)
    # concatenate all the string lists
    sentences = reduce(lambda x,y: x[0]+y[0], sentences)
    return zip(sentences, map(vectorize, [feature_space]*len(sentences),\
            sentences))

def centrality(similarity, vector, vectors):
    """ Computes centrality of a given vector regarding the given list of
    vectors and using the given similarity metric. """
    return 1.0/len(vectors)*sum([similarity(vector,y) for y in vectors\
            if y != vector])

def rank_by_centrality(collection_path, sim_func):
    """ Ranks the sentences from the collection in collection_path by their
    centrality. """
    # First get the feature space and vectorize the collection
    fs = create_collection_feature_space(collection_path)
    vectorized = vectorize_collection(fs, collection_path)
    all_vectors = [vector for (sent, vector) in vectorized]
    # Compute all centrality values
    centralities = [(sent, centrality(sim_func, vect, all_vectors))\
            for (sent, vect) in vectorized]
    # sort by centrality and return that list
    centralities.sort(key = lambda x: x[1], reverse=True)
    return centralities

def topic_weight(sentence, topic_words):
    """ Given a sentence and a list of topic words, returns the number of
    topic words in the sentence. """
    topic_words = set(topic_words) # slight speedup, lookup in set is faster
    topic_weight = 0
    for word in sentence.split():
        if word in topic_words:
            topic_weight += 1
    return topic_weight

def rank_by_tweight(collection_path, topic_file):
    """ Ranks the collection in collection_path by topic weight, using the
    topic words found in topic_file. """
    # First get the sentences and the topic words
    ts = load_topic_words(topic_file).keys()
    sentences = load_collection_sentences(collection_path, __fape_files_to_load)
    # reduce to 1-dimensional list of tuples (sentence, filename)
    sentences = reduce(lambda x,y: x+y, sentences)
    # Compute all topic weight values:
    tweights = [(sent, topic_weight(sent[0], ts)) for sent in sentences]
    # Sort descending and return
    tweights.sort(key = lambda x: x[1], reverse=True)
    return tweights

def combined_ranking(collection_path, topic_file, sim_func):
    """ Combines centrality and topic weight ranking into one. """
    cent_rank = rank_by_centrality(collection_path, sim_func)
    tweight_rank = rank_by_tweight(collection_path, topic_file)
    # Normalize topic weights to have Values in [0,1]
    max_weight = max([w for (x,w) in tweight_rank])
    tweight_rank = [(x,(float(w)/float(max_weight))) for (x,w) in tweight_rank]
    # Combine ranking by summing the weights. First sort rankings by sentences
    tweight_rank.sort(key=lambda x: x[0])
    cent_rank.sort(key=lambda x: x[0])
    # Now we have the sentence/rank pairs in the same order, so we can just sum
    # the corresponding ranks.
    combined_rank = []
    for ((s,w),i) in enumerate(tweight_rank):
        combined_rank.append((s,w + cent_rank[i][1]))
    # Sort the new ranking descending
    combined_rank.sort(key = lambda x: x[1], reverse = True)
    # Fin
    return combined_rank

# 2.3
def summarize_ranked_sentences(ranked_sents, summary_len):
    """ Returns a summary of approximately summary_len made from the list of
    ranked sentences in ranked_sents. """
    summary = []
    for (sent, value) in ranked_sents:
        summary.append(sent)
        summary_len -= len(sent.split())
        if summary_len <= 5:
            # We stop at n-5 words already, as an effort to avoid being
            # really far off.
            break
    return summary

def summarize_ranked_sentences_fixed(ranked_sents, summary_len,\
        min_sentence_length=None, max_sentence_length=None,\
        max_similarity=None, sim_func=None):
    """ Returns a summary of approximately summary_len made from the list of
    ranked sentences in ranked_sents. """
    summary = []
    # Feature space for ranked sentences, needed for similarity testing:
    fs = create_feature_space(map(lambda x:x[0], ranked_sents))
    for (sent, value) in ranked_sents:
        if ((min_sentence_length is None or\
                len(sent.split()) >= min_sentence_length) and\
                (max_sentence_length is None or\
                len(sent.split()) <= max_sentence_length)):
            # Only append, if similarity low enough
            if sim_func is not None:
                sent_v = vectorize(fs, sent)
                similarities = [sim_func(vectorize(fs,v), sent_v)\
                        for v in summary]
                if len(similarities)==0 or max_similarity >= max(similarities):
                    summary.append(sent)
                    summary_len -= len(sent.split())
                    if summary_len <= 5:
                        # We stop at n-5 words already, as an effort to avoid
                        # being really far off.
                        break
            else:
                summary.append(sent)
                summary_len -= len(sent.split())
                if summary_len <= 5:
                    # We stop at n-5 words already, as an effort to avoid being
                    # really far off.
                    break
    return summary

# Homework 1 Similarity functions
def intersect(A, B):
    return map(lambda (x, y): min(x, y), zip(A, B))

def union(A, B):
    return map(lambda (x, y): max(x, y), zip(A, B))

def sumf(A):
    return float(sum(A))

def dice_similarity(A, B):
    return (2.0 * sumf(intersect(A, B))) / (sumf(A) + sumf(B))

def jaccard_similarity(A, B):
    return sumf(intersect(A, B)) / sumf(union(A, B))

def cosine_similarity(A, B):
    return sumf(intersect(A, B)) / math.sqrt(sumf(A) * sumf(B))

# Additional stuff for writeup
# They make assumptions about my file structure, so they won't necessarily
# work in the turnin.
def make_graphs():
    for collection_name in os.listdir(__fape_collection_base_path):
        topic_file = 'topicWords/'+collection_name+'.ts'
        collection_path = __fape_collection_base_path+collection_name
        output_file = 'graphviz/'+collection_name+'.gr'
        visualize_collection_topics(topic_file, collection_path, output_file)
        os.system('/usr/bin/dot -Tpng '+output_file+' -o graphviz/'+\
            collection_name+'.png')

def make_summaries(mi=None,ma=None,ms=None,s=None):
    summ_dict = {}
    for collection_name in os.listdir(__fape_collection_base_path):
        collection_path = __fape_collection_base_path+collection_name
        topic_file = 'topicWords/'+collection_name+'.ts'
        ranked_centrality = rank_by_centrality(collection_path,\
                cosine_similarity)
        ranked_tweight = rank_by_tweight(collection_path, topic_file)
        summ_dict["cent_"+collection_name] =\
                summarize_ranked_sentences_fixed(ranked_centrality, 150,mi,ma,ms,s)
        summ_dict["twei_"+collection_name] =\
                summarize_ranked_sentences_fixed(ranked_tweight, 150,mi,ma,ms,s)
    return summ_dict

def produce_outfile():
    f = open('summs.txt', 'w')
    for c in os.listdir(__fape_collection_base_path):
        f.write("Centrality, "+c+":\n")
        f.write(str(summaries["cent_"+c])+"\n")
        f.write("Topic weight, "+c+":\n")
        f.write(str(summaries["twei_"+c])+"\n")
    f.close()
