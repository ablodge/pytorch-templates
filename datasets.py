import torch.utils.data as data
import torch
from collections import Counter
import re, random


class XORData(data.Dataset):

    def __init__(self):
        super().__init__()

    def __getitem__(self, index):
        random.seed(index)
        x1 = random.uniform(0, 1)
        x2 = random.uniform(0, 1)
        y = 1 if bool(x1<0.5) != bool(x2<0.5) else 0
        return {'x': torch.FloatTensor([x1,x2]),
                'y': torch.FloatTensor([y]),
                'readable': f'{x1>=0.5} xor {x2>=0.5} => {y>=0.5}'}

    def __len__(self):
        return int(1e10)

#
# class JapaneseMorphData(Data):
#     train_file = './jpn-morph-data/train.txt'
#     test_file = './jpn-morph-data/test.txt'
#
#     B_RE = re.compile('.\|')
#     Clean_RE = re.compile('(^\|(\s\|)*|(\s\|)*$)')
#
#     def __init__(self, file='./jpn-morph-data/train.txt', vocab=None, max_vocab_size=None, max_size=None):
#         super().__init__()
#         print('Loading Japanese Morphology Data')
#         if file=='train': file = JapaneseMorphData.train_file
#         if file=='test': file = JapaneseMorphData.test_file
#         self.data = []
#         self.data_readable = []
#         if not vocab:
#             self.vocab = Counter()
#             # get vocab
#             for x_text, y_text in tsv_iter(open(file, 'r', encoding='utf16'), max_lines=max_size):
#                 for ch in x_text:
#                     self.vocab[ch] += 1
#             self.vocab = [v for v,i in self.vocab.most_common(max_vocab_size)]
#         else:
#             self.vocab = vocab
#         # get data as torch tensors
#         for x_text, y_text in tsv_iter(open(file, 'r', encoding='utf16'), max_lines=max_size):
#             x, y = JapaneseMorphData.process_data(x_text, y_text, self.vocab)
#             if len(x) != len(y) or len(y)==0: continue
#             x = torch.LongTensor(x)
#             y = torch.FloatTensor(y)
#             self.data.append((x, y))
#             self.data_readable.append(x_text+'\t'+y_text)
#
#         print(f'{len(self.data)} Train Sentences')
#
#     def readable(self, index):
#         return self.data_readable[index]
#
#     @staticmethod
#     def process_data(x_text, y_text, vocab):
#         y_text = JapaneseMorphData.Clean_RE.sub('',y_text)
#
#         x = [vocab.index(ch) if ch in vocab else len(vocab) for ch in x_text]
#
#         y = JapaneseMorphData.B_RE.sub('|', y_text)
#         y = [1 if ch == '|' else 0 for ch in y]
#         if len(x) != len(y):
#             print('Warning')
#             print('X', x_text)
#             print('Y', y_text)
#         return x,y
#
#     def __getitem__(self, index):
#         return self.data[index]
#
#     def __len__(self):
#         return len(self.data)
#
#
# class ParenthesisData(Data):
#
#     def __init__(self):
#         super().__init__()
#
#     def __getitem__(self, index):
#         random.seed(index)
#         r = random.randint(0,1)
#         if r==0:
#             return torch.LongTensor([0 if x=='(' else 1 for x in self.get_positive_parens()]), torch.FloatTensor([1])
#         else:
#             return torch.LongTensor([0 if x=='(' else 1 for x in self.get_negative_parens()]), torch.FloatTensor([0])
#
#     def readable(self, index):
#         random.seed(index)
#         r = random.randint(0,1)
#         if r==0:
#             return self.get_positive_parens() + '=> 1'
#         else:
#             return self.get_negative_parens() + '=> 0'
#
#     def __len__(self):
#         return int(1e10)
#
#
#     def get_positive_parens(self):
#         outcomes = {0:'()',1:'(*)',2:'()()',3:'(*)()',4:'()(*)',5:'()()()',6:'(*)()()',7:'()()(*)',8:'()(*)()',9:'(*)',10:'**'}
#         s = '*'
#         while '*' in s:
#             r = random.randint(0, 10)
#             s = s.replace('*', outcomes[r], 1)
#         return s
#
#     def get_negative_parens(self):
#         s = self.get_positive_parens()
#         r = random.randint(0,len(s)-1)
#         s = s[:r]+s[r+1:]
#         return s


#
# def main():
#     test = [XORData(), ParenthesisData(), JapaneseMorphData('train',max_size=100)]
#
#     for t in test:
#         for i in range(100):
#             print(t.readable(i))
#             print(t[i])
#
#
# if __name__=='__main__':
#     main()


