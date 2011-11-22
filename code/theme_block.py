class Theme:
    def __init__(self):
        self.sentences = {}
        self.earliest = None
        self.most_informative = None

    def add_sentence(self, ((sent, date, filename, segment), weight)):
        self.sentences[sent] = weight;
        if self.earliest is not None:
            self.earliest = (sent,date)\
                    if date < self.earliest[1] else self.earliest
        else:
            self.earliest = (sent,date)
        if self.most_informative is not None:
            self.most_informative = (sent, weight)\
                    if self.most_informative[1] < weight\
                    else self.most_informative
        else:
            self.most_informative = (sent, weight)

    def __cmp__(self, other):
        if self.earliest[1] < other.earliest[1]:
            return -1
        elif self.earliest[1] == other.earliest[1]:
            return 0
        else:
            return 1

class Block():
    def __init__(self):
        self.themes = []
        self.earliest = None

    def add_theme(self, theme):
        self.themes.append(theme)
        if self.earliest is not None:
            self.earliest = theme if theme < self.earliest else self.earliest
        else:
            self.earliest = theme

    def __cmp__(self, other):
        if self.earliest < other.earliest:
            return -1
        elif self.earliest == other.earliest:
            return 0
        else:
            return 1
