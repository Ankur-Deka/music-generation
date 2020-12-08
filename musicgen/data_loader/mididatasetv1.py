from data_loader.datautils import *
from os import path as osp
from torch.utils.data import Dataset


class BasicMidiDataset(Dataset):

    def __init__(self, r_dir, vocab_file, train=True, crop=None, prefix=None, N=100):
        self.r_dir = osp.join(r_dir, 'train') if train else osp.join(r_dir, 'test')
        self.train = train
        self.files = []
        self.vocab = load_vocab(vocab_file)
        self.N = N
        print("Loaded vocabulary of size {}".format(len(self.vocab)))

        for r, dirs, files in os.walk(r_dir):
            files = list(filter(lambda x: x.endswith("mid") or x.endswith("midi"), files))
            if prefix is not None:
                files = list(filter(lambda x: x.startswith(prefix), files))
            files = list(map(lambda x: osp.join(r, x), files))
            self.files.extend(files)

        if crop is not None:
            self.files = self.files[:crop]

        # Set weights and weighing factor
        weights = [x[0] for x in self.vocab.values()]
        self.weighing_factor = np.mean(weights)*1.0
        self.weights = [0]*len(self.vocab)
        for k, v in self.vocab.items():
            wt, idx = v
            self.weights[idx] = self.weighing_factor/(1 + wt)

        # Load notes and chords
        self.notes = []
        discard = []
        self.count = []

        for i, f in enumerate(self.files):
            # Get notes
            notes = file_to_notes(f)[0]
            if len(notes) < self.N+1:
                discard.append(i)
                continue
            self.notes.append(notes)
            self.count.append(len(notes) - self.N - 1)

        # Create count
        if not train:
            self.count = [1]*len(self.count)
        print(self.count)
        self.length = sum(self.count)

    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        # From index figure out
        i = 0
        while idx >= self.count[i]:
            idx -= self.count[i]
            i += 1
        # Get sequence
        seq = self.notes[i][idx:idx+self.N]
        # Vocab is a dict of
        #  note or chord = [frequency, index]
        #
        # weights for sequence
        # Convert into indices
        seq = [self.vocab[x][1] for x in seq]

        out = self.notes[i][idx+self.N+1]
        out = self.vocab[out][1]
        return {
                'seq': torch.LongTensor(seq),
                'weight': torch.FloatTensor(self.weights),
                'out': torch.LongTensor([out]).squeeze(),
        }




if __name__ == '__main__':
    ds = BasicMidiDataset('data/Piano-midi.de/', 'data/albenizpianovocab.pkl', crop=5, prefix="alb")
    print(len(ds))
    print(ds[0])
    print(ds[0]['weight'].shape)
    # for k, v in ds[0].items():
        # print(k, v.shape)

    flag = True
    for i in range(len(ds)):
        try:
            k = ds[i]
        except Exception as e:
            flag = False
            print(i, e)
            print(ds.count)
            break
    if flag:
        print("No errors found")
