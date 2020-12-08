import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
from glob import glob
import mido
import string
from music21 import midi
from music21 import converter, instrument, note, chord
from itertools import groupby

# dataset
import os
from torch.utils.data import Dataset, DataLoader, random_split
import pickle as pkl

# tensorboard
from torch.utils.tensorboard import SummaryWriter


def load_vocab(vocabfile):
    # Load vocab dict from filename
    with open(vocabfile, 'rb') as fi:
        return pkl.load(fi)


def save_vocab(cdir, vocabfile, prefix=None):
    # Given a directory and output filename, create a vocabulary of notes and chords
    allfiles = []
    for r, dirs, files in os.walk(cdir):
        files = list(filter(lambda x: x.endswith('mid') or x.endswith('midi'), files))
        if prefix is not None:
            files = list(filter(lambda x: x.startswith(prefix), files))
        files = list(map(lambda x: os.path.join(r, x), files))
        allfiles.extend(files)

    allnotes = []
    for f in allfiles:
        print("Adding notes from {}".format(f))
        notes = file_to_notes(f)[0]
        allnotes.extend(notes)

    print("{} notes and chords found in total.".format(len(allnotes)))
    # allnotes = sorted(list(set(allnotes)))
    allnotes = sorted(allnotes)
    allnotes = [(x[0], len(list(x[1]))) for x in groupby(allnotes)]   # Keep list of notes and counts
    allnotes = [(x[0], (x[1], idx)) for idx, x in enumerate(allnotes)]

    # allnotes = [(k, i) for i, k in enumerate(allnotes)]
    allnotes = dict(allnotes)
    print("{} unique notes and chords found.".format(len(allnotes)))
    with open(vocabfile, 'wb') as fi:
        pkl.dump(allnotes, fi)
        print("Vocab saved to {}".format(vocabfile))


def file_to_notes(filename):
    # Given
    notes = []
    diff = []
    curtime = None

    midi = converter.parse(filename)
    notes_to_parse = None
    parts = instrument.partitionByInstrument(midi)
    if parts: # file has instrument parts
        partlen = [len(x.recurse()) for x in parts.parts]
        idx = np.argmax(partlen)
        notes_to_parse = parts.parts[idx].recurse()
    else: # file has notes in a flat structure
        notes_to_parse = midi.flat.notes

    for element in notes_to_parse:
        if curtime is None:
            if isinstance(element, note.Note) or isinstance(element, chord.Chord):
                curtime = float(element.offset)

        # print(element)
        if isinstance(element, note.Note):
            diff.append(float(element.offset) - curtime)
            notes.append(str(element.pitch))
            curtime = float(element.offset)
        elif isinstance(element, chord.Chord):
            diff.append(float(element.offset) - curtime)
            notes.append('.'.join(str(n) for n in element.normalOrder))
            curtime = float(element.offset)
    return notes, diff

