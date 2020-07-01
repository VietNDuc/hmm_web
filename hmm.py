import os
import re
import csv
import numpy as np


THRESHOLD = 1
SUFLEN = 2
MORPHCATNUM = 16
OPENCLASS = set([':','CD','FW','IN','JJ','JJR','JJS','NN',\
                 'NNP','NNPS','NNS','RB','RBR','RBS','UH',\
                 'VB','VBD','VBG','VBN','VBP','VBZ','SYM'])


def morphCat(word):
    # Words
    if re.match('\A[a-zA-Z]+\Z', word):
        # lower case
        if re.match('\A[a-z]+\Z', word):
            return 0
        # cap word
        elif re.match('\A[A-Z][a-z]*\Z', word):
            return 1
        # all cap
        elif re.match('\A[A-Z]+\Z', word):
            return 2
        else:
            return 3
    # hyphen
    elif '-' in word:
        # Cap-Cap
        if re.match('\A[A-Z][^-]*-[A-Z].*\Z', word):
            return 4
        # digit-seq
        elif re.match('\A\d{1,3}(,?\d{3})*(.\d*)?-.*\Z', word):
            return 5
        # seq-digit
        elif re.match('\A\D+-\d+\Z', word):
            return 6
        # lower seq - cap seq
        elif re.match('\A[a-z]+-[A-Z].*\Z', word):
            return 7
        # cap seq - lower seq
        elif re.match('\A[A-Z]+-[a-z].*\Z', word):
            return 15
        # include '-'
        elif word.count('-') > 1:
            return 8
        else:
            return  9
    # digits
    elif re.search('\d', word):
        #
        if re.match('\A[+-]?\d{1,3}(,?\d{3})*(.\d*)?\Z', word):
            return 10
        else:
            return 11
    elif '\/' in word:
        return 12
    elif '.' in word:
        return 13
    else:
        return 14


class HMM_Pos:
    """
    """
    def __init__(self):
        self.Pemit = {}     # { Pos : { word : Pemit }}
        self.Words = {}     # { word : { Pos : count} ) }
        self.PosSize = 0
        self.suffix = {}


    def _load_prepare(self, dataPath):
        """
        """
        try:
            ftrain = open(os.path.join(dataPath, 'WSJ_02-21.pos'), 'r')
        except:
            print("Invalid path")
            exit()

        self.Ptrans = {'START': {}}  #{ (Pos, Pos): {Pos : count}}
        PosPre = 'START'  # state t-1
        PosPP = ''  # state t-2

        for line in csv.reader(ftrain, delimiter='\t'):
            if len(line) == 0:
                Pos = 'END'
                self.Ptrans[(PosPP, PosPre)][Pos] = self.Ptrans.setdefault((PosPP, PosPre), {Pos: 1}).get(Pos, 0) + 1
                PosPre = 'START'
                continue

            word, Pos = line[0], line[1]

            # word POS
            self.Words[word][Pos] = self.Words.setdefault(word, {Pos: 1}).get(Pos, 0) + 1

            # emission count
            self.Pemit[Pos][word] = self.Pemit.setdefault(Pos, {word: 1}).get(word, 0) + 1

            # transition count
            if PosPre == 'START':
                self.Ptrans['START'][Pos] = self.Ptrans['START'].get(Pos, 0) + 1
            else:
                self.Ptrans[(PosPP,PosPre)][Pos] = self.Ptrans.setdefault((PosPP,PosPre), {Pos: 1}).get(Pos, 0) + 1

            PosPP = PosPre
            PosPre = Pos
        ftrain.close()

        self.PosSize = len(self.Pemit)
        self.label = {Pos: enum for enum, Pos in enumerate(self.Pemit)}
        tmp = [(self.label[Pos], Pos) for Pos in self.label]
        tmp.sort()
        self.tag = [t[1] for t in tmp]
        self.tag.append('START')
        self.label.update({'END': self.PosSize,'START': self.PosSize})

    def _suffix(self):
        """
        """
        self.suffix = {}
        for word in self.Words:
            numsuf = SUFLEN if len(word) >= SUFLEN else len(word)
            sufl = [word[-i:] for i in range (1, numsuf+1)]
            for suf in sufl:
                if suf not in self.suffix:
                    self.suffix[suf] = np.zeros(self.PosSize + 1)
                for Pos in self.Words[word].keys():
                    self.suffix[suf][self.label[Pos]] += 1

        for suf in list(self.suffix):
            total = np.sum(self.suffix[suf])
            if total >= 5:
                self.suffix[suf] += 1
                self.suffix[suf] *= 1. / np.sum(self.suffix[suf])
            else:
                self.suffix.pop(suf)

    def _morph(self):
        """
        """
        self.morph = np.zeros([MORPHCATNUM, self.PosSize+1])

        for word in self.Words:
            cat = morphCat(word)
            for Pos in self.Words[word].keys():
                if cat > 1:
                    self.morph[cat, self.label[Pos]] += 1
                else:
                    if self.Words[word][Pos] <= THRESHOLD:
                        self.morph[cat, self.label[Pos]] += 1

        self.morph += 1
        self.morph = 1. / np.sum(self.morph, axis=1)

    def _get_trans_matrix(self):
        """
        """
        self.TransMat = np.zeros([self.PosSize+1, self.PosSize+1, self.PosSize+1])

        for PosPre in self.Ptrans:
            if PosPre == 'START':
                i = self.label['START']
                j = i
            else:
                i = self.label[PosPre[0]]
                j = self.label[PosPre[1]]
            for Pos in self.Ptrans[PosPre]:
                self.TransMat[i, j, self.label[Pos]] = self.Ptrans[PosPre][Pos]

        self.TransMat2 = np.sum(self.TransMat, axis=0)
        self.TransMat += 1
        self.TransMat = self.TransMat * 1. / np.sum(self.TransMat, axis=2, keepdims=True)

        self.TransMat2 *= 1. / np.sum(self.TransMat2, axis=1, keepdims=True)
        self.TransMat2 = self.TransMat2 * np.ones([self.PosSize + 1, self.PosSize + 1, self.PosSize + 1])

    def _get_pemit_matrix(self):
        self.unk = np.zeros(self.PosSize+1)

        for Pos in self.Pemit:
            vec = np.array(list(self.Pemit[Pos].values()))
            total = np.sum(vec)
            if Pos in OPENCLASS:
                self.unk[self.label[Pos]] = np.sum(vec <= THRESHOLD) * 1. / total
            for word in self.Pemit[Pos]:
                self.Pemit[Pos][word] *= 1. / total

        self.unk *= 1. / np.sum(self.unk)

        isg = self.TransMat[1:, :-1] >= self.TransMat2[1:, :-1]
        self.lambd3 = np.sum(self.TransMat[1:, :-1][isg])
        self.lambd2 = np.sum(self.TransMat2[1:, :-1][isg is not True])
        total = self.lambd2 + self.lambd3
        self.lambd3 /= total

    def _get_emission_unk_word(self, word):
        """
        """
        ret = []
        if word in self.Words:
            for Pos in self.Words[word].keys():
                ret.append((Pos, self.Pemit[Pos][word]))
        else:
            cat = morphCat(word)
            flag = 0
            numsuf = SUFLEN if len(word) >= SUFLEN else len(word)
            sufl = [word[-i:] for i in range(numsuf, 0, -1)]
            for suf in sufl:
                if suf in self.suffix:
                    ret = [(Pos, emit) for (Pos, emit) in zip(self.tag, self.unk * self.suffix[suf] * self.morph[cat])]
                    flag = 1
                    break
            if flag == 0:
                ret = [(Pos, emit) for (Pos, emit) in zip(self.tag, self.unk * self.morph[cat])]

            if cat < 4:
                if word.lower() in self.Words:
                    total = sum(self.Words[word.lower()].values())
                    for Pos in self.Words[word.lower()]:
                        ret[self.label[Pos]] = (Pos, ret[self.label[Pos]][1] + self.Words[word.lower()][Pos] * 1. / total)

        return ret

    def _Viterbi(self, phrase):
        """Viterbi algorithm
        """

        T = len(phrase)
        Vtb = np.zeros([T + 2, self.PosSize + 1, self.PosSize + 1])
        Trace = np.ones([T + 2, self.PosSize + 1, self.PosSize + 1]) * -1
        Vtb[0, self.label['START'], :] += 1
        ret = []

        # Recursive
        for i in range(1, T+1):
            word = phrase[i-1]
            PosSet = self._get_emission_unk_word(word)
            for Pos, emit in PosSet:
                tmp = Vtb[i-1] * (self.lambd3 * self.TransMat[:, :, self.label[Pos]] + self.lambd2 * self.TransMat2[: , :, self.label[Pos]])
                Vtb[i, :, self.label[Pos]] = np.max(tmp, axis=0) * emit *100
                Trace[i, :, self.label[Pos]] = np.argmax(tmp, axis=0)

        # END
        i = T+1
        Pos = 'END'
        tmp = Vtb[i-1] * (self.lambd3 * self.TransMat[:, :, self.label[Pos]] + self.lambd2 * self.TransMat2[:, :, self.label[Pos]])
        Vtb[i, :, self.label[Pos]] = np.max(tmp, axis=0)
        Trace[i, :, self.label[Pos]] = np.argmax(tmp, axis=0)
        ToPos = self.label['END']
        FromPos = int(np.argmax(Vtb[i, :, ToPos]))
        PrePos = int(Trace[i, FromPos, ToPos])
        for i in range(T, 0, -1):
            ret.append(self.tag[FromPos])
            ToPos = FromPos
            FromPos = PrePos
            PrePos = int(Trace[i, FromPos, ToPos])
        ret.reverse()

        return ret

    def train(self, dataPath):
        self._load_prepare(dataPath)
        self._suffix()
        self._morph()
        self._get_trans_matrix()
        self._get_pemit_matrix()

    def getTag(self, sentence):
        regex = r'[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+|[,.?]'
        phrase = re.findall(regex, sentence)
        tagger = self._Viterbi(phrase)
        rs = []
        for (x, y) in zip(phrase, tagger):
            rs.append('%s\t%s' % (x, y))
        return rs.join('')

