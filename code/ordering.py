####################################
# CIS 530 Project                  #
# Ordering module                  #
# Authors:                         #
# Quan Dong (qdong)                #
# Fabian Peternek (fape)           #
# Yayang Tian (yaytian)            #
####################################

from datetime import date
from theme_block import Theme, Block
from get_relatedness import get_relatedness
from sentence_selection import load_collection_sentences

import os

def extract_dates(sentence_list):
    """ Given a set of ranked sentences and their filenames extracts the dates
        from the filenames.
        sentence_list is expected to have the format 
            ((sentence, filename), weight)
        and the filenames look like 'ABCYYYYMMDD.xxxx.clean'.
        This function returns the sentences in the same order as it receives but
        includes a date object in additon to the filename.
    """
    output_list = []
    for ((sent, f_name, segment), w) in sentence_list:
        date_string = fname.split('.')[0][3:]
        year = int(date_string[:4])
        month = int(date_string[4:6])
        day = int(date_string[6:8])
        output_list.append(((sent, date(year,month,day), f_name, segment), w))
    return output_list

def strip_weights(sentence_list):
    """ Strips the weights from the sentence list, as the order is already
        descending by weight the actual weights shouldn't be needed anymore.
        Not done in extract_dates, because I might be wrong there.
    """
    output_list = []
    for ((sent, date), w) in sentence_list:
        output_list.append((sent, date))
    return output_list

def make_summary(sentence_list, max_similarity, sim_func, order_func):
    """ Given the ranked sentences makes the full summary. """
    pass

def chronological_ordering(themes):
    """ Applies the chronological ordering algorithm on the given set of themes, 
        This function basically returns the finished summary as it returns just
        the sentences representing the themes.
    """
    # Sort by date of publishment
    themes.sort()
    # return the summary sentences
    return [thm.most_informative[0] for thm in themes]

def augmented_ordering(themes):
    """ Applies the augmented algorithm on the given themes, so that
        the finished summary is returned as it returns just the most informative
        sentence for every theme.
    """
    # First compute ratio of relatedness graph for the themes 
    graph = create_relatedness_graph(themes)
    # compute transitive closure of the graph
    graph = transitive_closure(graph)
    # Get the connected components which we'll use as the blocks
    components = compute_components(graph)
    blocks = []
    for component in components:
        block = Block()
        for theme in component:
            block.add_theme(theme)
        blocks.append(block)
    # order the blocks chronologically ascending.
    blocks.sort()
    # order the themes in every block chronologically ascending
    for block in blocks:
        for theme in block.themes:
            theme.sort()
    # Finally get the most informative sentences
    return [thm.most_informative[0] for block in blocks for thm in block.themes]

def transitive_closure(graph):
    """ Computes the transitive closure of a given graph. 
        The Algorithm is straight-forward and fairly inefficient, but
        as our graphs shouldn't get very large, that probably won't matter.
        Just compute all possible paths from every node and add the new edges.
        graph is given as tuple of two lists: One containing the vertices and
        one containing the edges.
        Algorithm first computes the connected components of the graph by BFS
        and then basically just makes cliques out of every component.
        Complexity should be somewhere around O(VE+V^2).
    """
    # Compute components and put use their carthesian product as edges
    components = compute_components(graph)
    V = set(graph[0])
    E = set()
    for comp in components:
        E = E.union(set([frozenset([u,v]) for u in comp for v in comp\
                if u != v]))
    return (list(V), [tuple(e) for e in E])

def exists_path(start, end, (V,E)):
    """ Given start and end vertex, checks if there is a path from start to end
        in the given Graph. 
        Algorithm uses BFS.
    """
    return end in get_reachable_vertices(start, (V,E)) if start != end\
            else False

def compute_components(graph):
    """ Computes the connected components of the given graph and returns them as
        a list of vertexsets.
    """
    # Setify the graph
    V = set(graph[0])
    E = set([frozenset(edge) for edge in graph[1]])
    components = []
    # While unprocessed vertices exist
    while len(V) != 0:
        # Take any vertex and compute the connected component it belongs to
        v = iter(V).next()
        reachable = get_reachable_vertices(v, (V,E))
        components.append(reachable)
        V = V.difference(reachable)
    return components

def get_reachable_vertices(node, (V,E)):
    """ Implements BFS to find connected component. """
    to_visit = [u for u in V if frozenset([node,u]) in E]
    seen = set([node])
    while len(to_visit) != 0:
        v = to_visit.pop()
        if v not in seen:
            # get new neighborhood and add v to the component
            to_visit = [u for u in V if frozenset([v,u]) in E] + to_visit
            seen.add(v)
    return seen

def compute_similarity_matrix(vectors, sim_func, out_file):
    """ Computes pairwise similarities of all sentences represented by the
        featurespace using sim_func as similarity metric. Writes a matrix that 
        can be used as input for Cluto clustering into out_file.
    """
    outstring = str(len(vectors)) + "\n"
    for s1 in vectors:
        for s2 in vectors:
            outstring += str(sim_func(s1,s2)) + " "
        outstring += "\n"
    # Similarity matrix computed, write to file:
    f = open(out_file, 'w')
    f.write(outstring)
    f.close()

def cluster_sentences(similarity_matrix_file, cluto_bin, num_clusters=5):
    """ Uses Cluto to produce a clustering of the given similarity matrix.
        Returns the clustering vector.
    """
    clustfile = "./clusters/"+similarity_matrix_file+"."+num_clusters
    os.system(cluto_bin + "-clustfile=" + clustfile + similarity_matrix_file + " " + num_clusters)
    # now get the clusters
    f = open(clustfile, 'r')
    clusters = f.readlines()
    f.close()
    return clusters

def make_themes_from_clusters(sentences, clusters):
    """ Given a set of sentences and the clusters they belong to
        constructs a list of themes.
        Sentences is a list of tuples, that have the following form:
        ((sentence, date, filename), topic_weight)
    """
    # First create empty theme for every cluster
    themes = [Theme() for i in set(clusters)]
    # Now add every sentence into the cluster/theme it belongs to
    for (i, sent) in enumerate(sentences):
        themes[clusters[i]].add_sentence(sent)
    return themes

def create_relatedness_graph(themes):
    """ Computes pairwise relatedness of the themes and returns a
        graph, that has the themes as nodes and an edge between two
        nodes, if the themes' relatedness is >=0.6
    """
    V = themes
    E = []
    for t1 in themes:
        for t2 in themes:
            if t1 != t2: # We don't want reflexive edges
                if get_relatedness(t1,t2) >= 0.6:
                    E.append((t1,t2))
    return (V,E)

def augmented_preprocessing(collection_path):
    """ Loads the texts from the given collection and computes the
        topic weight for each of them by computing the topic words
        first.
        Returns a list of themes, which can then be ordered.
    """

