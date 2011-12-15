#from theme_block import Theme # Not even necessary

def get_relatedness(theme1,theme2):
    nAB=0
    nAB_plus=0
    for sentence_1 in theme1.sentences:
        for sentence_2 in theme2.sentences:
            if cmp(sentence_1[2],sentence_2][2])==0:
                nAB=nAB+1
                if sentence_1[3]==sentence_2[3]:
                    nAB_plus=nAB_plus+1
    if nAB==0:
        return 0
    else:
        return float(nAB_plus)/float(nAB)
