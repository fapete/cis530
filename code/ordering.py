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

