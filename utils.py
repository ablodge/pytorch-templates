import time, string, os

def describe(tensor):
    print(f'Type: {tensor.type()}')
    print(f'Shape: {tensor.size()}')
    print(f'Values: {tensor}')


def tsv_iter(file_reader, max_lines=-1):
    progress = ProgressBar(f'Read File {file_reader}', max_lines)
    progress.start_timer()

    for i, line in enumerate(file_reader):
        if max_lines==-1 or i >= max_lines:
            break
        if i%1000 == 0:
            progress.report(i)
        yield line[:-1].split('\t')
    progress.end_timer()


def dir_iter(dir, file_extension='.txt'):
    progress = ProgressBar(f'Loading Files {dir}', len([f for f in os.listdir(dir) if f.endswith(file_extension)]))
    for i, filename in enumerate(os.listdir(dir)):
        if filename.endswith(file_extension):
            with open(os.path.join(dir,filename),'r', encoding='utf8') as f:
                yield f.read()
        progress.report(i+1)
    progress.end_timer()


class ProgressBar:

    def __init__(self, name, total_size):
        self.name = name
        self.start = time.time()
        self.total_size = total_size

    def start_timer(self):
        self.start = time.time()

    def report(self, progress, also=''):
        end = time.time()
        eta = (self.total_size - progress) * (end - self.start) / progress
        eta_h = int(eta / 3600)
        eta_m = '{:0>2d}'.format(int((eta % 3600) / 60))
        eta_s = '{:0>2d}'.format(int(eta % 60))
        print(f'\r{self.name} {progress}/{self.total_size} ETA {eta_h}h:{eta_m}m:{eta_s}s {also}', end='')

    def end_timer(self, also=''):
        end = time.time()
        eta = (end - self.start)
        eta_h = int(eta / 3600)
        eta_m = '{:0>2d}'.format(int((eta % 3600) / 60))
        eta_s = '{:0>2d}'.format(int(eta % 60))
        print(f'\r{self.name} {self.total_size}/{self.total_size} Finished {eta_h}h:{eta_m}m:{eta_s}s {also}')


class SequenceVocabulary:

    MASK = '[MASK]'
    START = '[START]'
    END = '[END]'
    UNK = '[UNK]'
    PUNCTUATION = string.punctuation

    def __init__(self):
        self.vocab_index = []
        for t in [self.MASK, self.START, self.END, self.UNK]:
            self.add(t)

    def add(self, tok):
        self.vocab_index.append(tok)

    def index(self, tok):
        if not tok in self.vocab_index:
            return self.vocab_index.index(self.UNK)
        return self.vocab_index.index(tok)

    def read(self, tokens):
        if isinstance(tokens, int):
            return self.vocab_index[index]
        elif isinstance(tokens, list):
            return ' '.join(self.vocab_index[i] for i in tokens)

    def sort(self):
        self.vocab_index.sort()

    def pad(self, seq, total_length):
        while len(seq) < total_length:
            seq.append(self.index(self.MASK))

    def __len__(self):
        return len(self.vocab_index)