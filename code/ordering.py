####################################
# CIS 530 Project                  #
# Ordering module                  #
# Authors:                         #
# Quan Dong (qdong)                #
# Fabian Peternek (fape)           #
# Yayang Tian (yaytian)            #
####################################

from datetime import date

def extract_dates(sentence_list):
    """ Given a set of ranked sentences and their filenames extracts the dates
        from the filenames.
        sentence_list is expected to have the format 
            ((sentence, filename), weight)
        and the filenames look like 'ABCYYYYMMDD.xxxx.clean'.
        This function returns the sentences in the same order as it receives but
        replaces the filename with a date object.
    """
    output_list = []
    for ((sent, f_name), w) in sentence_list:
        date_string = fname.split('.')[0][3:]
        year = int(date_string[:4])
        month = int(date_string[4:6])
        day = int(date_string[6:8])
        output_list.append(((sent, date(year,month,day)), w))
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

def chronological_ordering(sentence_list):
    """ Applies the chronological ordering algorithm on the given sentences, 
        which should already make up a summary. The dates will be stripped from
        the list in the process such that this function basically returns the
        finished summary.
    """
    # Sort by date of publishment
    sentence_list.sort(key = lambda x: x[1])
    # Strip out the dates, no longer needed
    return [sent for (sent, date) in sentence_list]

def augmented_ordering(sentence_list):
    """ Applies the augmented algorithm on the given sentences, which should
        already make up a summary. Strips the dates from the sentences, so that
        the finished summary is returned.
    """
    pass

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
