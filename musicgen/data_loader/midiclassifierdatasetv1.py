from data_loader.datautils import *
from os import path as osp
from torch.utils.data import Dataset
from music21 import midi, stream
from music21 import converter, instrument, note, chord
from itertools import groupby

class MidiClassifierDataset(Dataset):

    def __init__(self, r_dir, vocab_file, train=True, crop=None, prefix=None, N=100):
        self.r_dir = osp.join(r_dir, 'train') if train else osp.join(r_dir, 'test')
        self.train = train
        self.files = []
        self.vocab = load_vocab(vocab_file)
        self.N = N
        print("Loaded vocabulary of size {}".format(len(self.vocab)))

        for r, dirs, files in os.walk(self.r_dir):
            files = list(filter(lambda x: x.endswith("mid") or x.endswith("midi"), files))
            if prefix is not None:
                if isinstance(prefix, str):
                    files = list(filter(lambda x: x.startswith(prefix), files))
                elif isinstance(prefix, list):
                    files = list(filter(lambda x: any([x.startswith(y) for y in prefix]), files))
                else:
                    raise NotImplementedError

            files = list(map(lambda x: osp.join(r, x), files))
            self.files.extend(files)

        if crop is not None:
            self.files = self.files[:crop]

        # Get classes
        artists = [x.split('/')[-1].split('_')[0] for x in self.files]
        artists = sorted(list(set(artists)))
        # Get index, sorted list, and counts
        self.artistsidx = dict([(v, i) for i, v in enumerate(artists)])
        self.artistslist = artists
        self.artistcount = dict([(v, 0) for v in artists])

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
            # Also count this towards artist count
            self.artistcount[self.get_artist_name(f)] += len(notes) - self.N - 1

        #print(self.artistcount)
        #print(self.artistslist)
        self.weights = np.array([1./self.artistcount[x] for x in self.artistslist])
        self.weights = self.weights/self.weights.sum()
        #print(self.weights)

        # Create count
        if not train:
            self.count = [1]*len(self.count)
        print(self.count)
        self.length = sum(self.count)
        self.get_reverse_vocab()


    def get_artist_name(self, filename):
        return filename.split('/')[-1].split('_')[0]

    def get_reverse_vocab(self, ):
        self.revvocab = dict()
        for k, v in self.vocab.items():
            self.revvocab[v[1]] = k

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

        out = self.get_artist_name(self.files[i])
        out = self.artistsidx[out]

        # Get next note too
        nextnote = self.notes[i][idx+self.N]
        nextnote = self.vocab[nextnote][1]

        return {
                'seq': torch.LongTensor(seq),
                'weight': torch.FloatTensor(self.weights),
                'out': torch.LongTensor([out]).squeeze(),
                'nextnote': torch.LongTensor([nextnote]).squeeze(),
        }

    def convert_to_midi(self, notes):
        # given array of notes, convert to notes and then to midi
        B, S = notes.shape
        for i in range(B):
            # parse a sequence
            melody = []
            offset = 0
            for s in range(S):
                # Something like A#
                token = self.revvocab[notes[i, s]]
                if '.' in token or token.isdigit():
                    new_note = token.split('.')
                    new_note = [note.Note(int(x)) for x in new_note]
                    for _ in range(len(new_note)):
                        new_note[_].storedInstrument = instrument.Piano()
                    new_chord = chord.Chord(new_note)
                    new_chord.offset = offset
                    melody.append(new_chord)
                else:
                    # note
                    new_note = note.Note(token)
                    new_note.offset = offset
                    new_note.storedInstrument = instrument.Piano()
                    melody.append(new_note)
                offset += 0.5
            # Parsed entire sequence, save it
            midistream = stream.Stream(melody)
            filename = self.files[i].split('/')[-1]
            filename = filename.replace('.mid', '_output.mid')
            midistream.write('midi', filename)
            print("Written to {}".format(filename))




if __name__ == '__main__':
    #ds = MidiClassifierDataset('data/Piano-midi.de/', 'data/albenizpianovocab.pkl', crop=5, prefix="alb")
    ds = MidiClassifierDataset('data/Piano-midi.de/', 'data/pianovocab.pkl', prefix=['alb', 'mendel', 'muss'])
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
