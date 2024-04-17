# Copyright (c) 2024 Jingjing Tang
#
# -*- coding:utf-8 -*-
# @Script: data_preprocessing.py
# @Author: Jingjing Tang
# @Email: tangjingjingbetsy@gmail.com
# @Create At: 2024-04-15 17:02:52
# @Last Modified By: Jingjing Tang
# @Last Modified At: 2024-04-15 17:02:52
# @Description: This is data_processing for transcribed scores.

from tokenizers.octuple_performer import OctuplePerformer
from miditoolkit import MidiFile
from config import *
import pprint, os, copy
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from pretty_midi import PrettyMIDI
import itertools
import sys

NUM_OF_SCALES = 3
PAD_ID = 0
EOS_ID = 2
SOS_ID = 1
MAX_LEN = 1000
VOCAB_CATEGORIES = [
    "pitch",
    "velocity",
    "duration_dev",
    "position",
    "bar",
    "performer",
    "ioi"
]
NUM_OF_CATEGORIES = len(VOCAB_CATEGORIES)
NUM_OF_PERFORMERS = 49
BEATS_PER_BAR = 4
TICS_PER_BEAT = 384


class Tokenizer():
    """
    Self-defined Tokenizer Object for easy usage base on MidTok Tokenizer
    """
    def __init__(self, 
                 res=TICS_PER_BEAT, 
                 path_to_tokenizer=None,
                 quantization=None,
                 withPedal=False, 
                 withComposition=False,
                 with_sos_eos=False):
        """initialize a tokenizer

        Args:
            res (int, optional): resolution of the midi files. Defaults to 384.
            path_to_tokenizer (str|None, optional): path to a save tokenizer. Defaults to None.
            quantization (str| None, optional): type of quantization used to process the midi. Defaults to None.
            withPedal (bool, optional): whether to consider pedalling when tokenizing. Defaults to False.
            withComposition (bool, optional): whether to tokenize composition id. Defaults to False.
            with_sos_eos (bool, optional): whether to add sos and eos tokens. Defaults to False.
        """
        self.with_sos_eos = with_sos_eos

        if path_to_tokenizer != None:
            self.tokenizer = self.load(path_to_tokenizer)
            
        else:
            pitch_range = range(21,109)
            beat_res = {(0, 12):res}
            nb_velocities = 64
            additional_tockens = {'Tempo': False, 
                                'TimeSignature': False,
                                'Chord': False,
                                'Rest': False,
                                'Program': False,
                                'Pedal': withPedal,
                                'Composition': withComposition,
                                'nb_tempos': 64,  # nb of tempo bins
                                'tempo_range': (40, 250)}  # (min, max)

            self.tokenizer = OctuplePerformer(pitch_range, 
                                        beat_res, 
                                        nb_velocities, 
                                        additional_tockens, 
                                        mask=False, 
                                        sos_eos_tokens=with_sos_eos, 
                                        num_of_performer=NUM_OF_PERFORMERS, is_quantize=quantization)
            
    @staticmethod
    def load(path_to_tokenizer):
        """Load a saved tokenizer

        Args:
            path_to_tokenizer (str): path to the save file

        Returns:
            MIDITokenizer: a tokenizer object defined in MidTok
        """
        tokenizer = np.load(path_to_tokenizer, allow_pickle=True).item()
        return tokenizer
    
    def token2midi(self, tokens, path_to_midi):
        """transfer the tokens to a midi file

        Args:
            tokens (list): list of tokens
            path_to_midi (str): path to the save directionary
        """
        midi_rep = self.tokenizer.tokens_to_midi([tokens])
        midi_rep.dump(filename=path_to_midi)
        return
    
    def token2event(self, tokens):
        """convert tokens to events

        Args:
            tokens (list): list of token values

        Returns:
            numpy array: array of events
        """
        events = self.tokenizer.tokens_to_events(tokens)
        return np.asarray(events).squeeze()
    
    def tokenize_midi(self, midi_file, performer):
        midi_file = MidiFile(midi_file)
        tokens = self.tokenizer.midi_to_tokens(midi_file, performer)
        return tokens
    
    def tokenize_midi_file(self, path_to_midi, performer_id):
        """tokenize one midi file

        Args:
            path_to_midi (str): path to the midi file
            performer_id (int): identity of the performer

        Returns:
            array: array of token values
        """
        tokens = self.tokenize_midi(path_to_midi, performer_id)
        return np.asarray(tokens).squeeze()

class CreateTokenDataset():
    """
    Create dataset of token values for training
    """
    def __init__(self, 
                 path_to_data_csv=str,
                 save_path_of_data=str,
                 save_path_of_tokenizer=str,
                 tokenizer=Tokenizer,
                 data_folder_path=None,
                 max_len=MAX_LEN,
                 number_of_categories=NUM_OF_CATEGORIES,
                 isPair=False,
                 isAugument=False):
        
        self.df = pd.read_csv(path_to_data_csv)
        
        if isPair:
            self.performance_folder = data_folder_path[0]
            self.score_folder = data_folder_path[1]
            
        else:
            self.data_folder = data_folder_path[0]
            
        self.tokenizer = tokenizer
        self.isPair = isPair
        self.isAugument = isAugument
        self.max_len = max_len
        self.number_of_categories = number_of_categories
        self.save_path = save_path_of_data
        self.save_path_tokenizer = save_path_of_tokenizer
            
        self.x_list = list()
        self.y_list = list()
        self.performer_list = list()
        self.mask_list = list()
        self.piece_name_list = list()
    
    @staticmethod
    def segment_sequences_with_attention_masks(input_seqs, max_len, number_of_categories):
        x_list = list()
        mask_list = list()
        
        seq_len = len(input_seqs)
            
        if seq_len <= max_len:
            tmp = copy.deepcopy(input_seqs)
            x = np.concatenate([tmp, np.ones((max_len-seq_len, number_of_categories)) * PAD_ID])
            attn_mask = np.concatenate([np.ones(seq_len), np.zeros(max_len-seq_len)])
            x_list.append(x)
            mask_list.append(attn_mask)    
        
        else:
            start_index = 0
            tmp = copy.deepcopy(input_seqs)
            while start_index + max_len <= seq_len - 1:
                # overlap_length = np.random.randint(50, 100)
                overlap_length = 50
                x = tmp[start_index:start_index + max_len]
                start_index += max_len - overlap_length    
                attn_mask = np.ones(max_len)
                
                x_list.append(x)
                mask_list.append(attn_mask)                
            
            try:
                x = tmp[start_index:]
                end_len = len(x)
                
                x = np.concatenate([x, np.ones((max_len-end_len, number_of_categories)) * PAD_ID])
                attn_mask = np.concatenate([np.ones(end_len), np.zeros(max_len-end_len)])
            
            except:
                print("fail to segment input sequences")
                raise ValueError
            
            x_list.append(x)
            mask_list.append(attn_mask)

        return x_list, mask_list
        
    @staticmethod
    def align_p_and_s_tokens(ref_token, target_token):
        target_list = []
        extra = []
        for i in range(len(ref_token)):
            if i > len(target_token):
                extra.append(i)
                continue
            is_find = False
            
            if i <= 10:
                start = 0
            else:
                start = i - 10
                
            if i <= len(target_token) - 20:
                end = i + 20    
            else:
                end = len(target_token)
                
            for j in range(start, end):
                if (ref_token[i][0] == target_token[j][0]) and \
                (ref_token[i][1] == target_token[j][1]):
                    if len(target_list) > 0:
                        if ref_token[i][4] - ref_token[i-1][4] <= 1:
                            if target_token[j][4] * 4 * TICS_PER_BEAT + target_token[j][3]- \
                                target_list[-1][4] * 4 * TICS_PER_BEAT  - target_list[-1][3] > - 4 * TICS_PER_BEAT :
                                    target_list.append(target_token[j])
                                    is_find = True
                                    break
                        else:
                            if (target_token[j][4] - target_list[-1][4] > 1) and \
                                (target_token[j][4] - target_list[-1][4] < 10):
                                target_list.append(target_token[j])
                                is_find = True
                                break
                    else:
                        target_list.append(target_token[j])
                        is_find = True
                        break
                
            if is_find == False:
                extra.append(i)

        ref_list = [i for j, i in enumerate(ref_token) if j not in extra]
        
        if len(ref_list) == len(target_list):
            if len(extra) > 0:
                print(len(extra))
            return ref_list, target_list
        else:
            print("fail to align two sequences")
            raise ValueError
        
    def create_segments_for_token_sequences(self, input_seqs):
        for i in range(2):
            seqs = input_seqs[i]
            if len(seqs[0]) < self.number_of_categories:
                input_seqs[i] = np.concatenate([seqs, 
                                        np.zeros((len(seqs), self.number_of_categories-len(seqs[0])))], 
                                        axis=1)
            
        if "ioi" in VOCAB_CATEGORIES:
            ioi_index = VOCAB_CATEGORIES.index("ioi")
            pos_index = VOCAB_CATEGORIES.index("position")
            bar_index = VOCAB_CATEGORIES.index("bar")
            
            for i in range(len(input_seqs[0])):
                if i < len(input_seqs[0]) - 1:
                    for j in range(2):
                        seqs = input_seqs[j]
                        input_seqs[j][i][ioi_index] = seqs[i+1][bar_index]* 4 * TICS_PER_BEAT  + seqs[i+1][pos_index] -\
                                                (seqs[i][bar_index]* 4 * TICS_PER_BEAT  + seqs[i][pos_index])
        
        if "duration_dev" in VOCAB_CATEGORIES:
            dur_index = VOCAB_CATEGORIES.index("duration_dev")
            for i in range(len(input_seqs[0])):
                if i < len(input_seqs[0]) - 1:
                    input_seqs[0][i][dur_index] -= input_seqs[1][i][dur_index]
        
                                            
        if self.tokenizer.with_sos_eos:
            input_seqs = np.concatenate([
                np.ones((1, self.number_of_categories)) * SOS_ID, 
                input_seqs, 
                np.ones((1, self.number_of_categories)) * EOS_ID]) 
        
        
        if self.isPair:
            x_list, mask_list = self.segment_sequences_with_attention_masks(input_seqs[1], 
                                                                        self.max_len, 
                                                                        self.number_of_categories)
            self.x_list += x_list
            self.mask_list += mask_list
            
            y_list, _ = self.segment_sequences_with_attention_masks(input_seqs[0],
                                                                 self.max_len, 
                                                                 self.number_of_categories)
            self.y_list += y_list  
            
            assert len(self.x_list) == len(self.y_list)  
        else:
            x_list, mask_list = self.segment_sequences_with_attention_masks(input_seqs[0], 
                                                                         self.max_len, 
                                                                         self.number_of_categories)
            self.x_list += x_list
            self.mask_list += mask_list
        
        performer_list = list()
        piece_name_list = list()
        for i in range(len(x_list)):
            performer_list.append(self.row['artist_id'])
            piece_name_list.append(self.row['midi_path'])
        
        self.performer_list += performer_list
        self.piece_name_list += piece_name_list
    
    def process(self):
        for idx, row in tqdm(self.df.iterrows(), 
                         total= self.df.shape[0], 
                         bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:20}{r_bar}'):
            self.row = row
    
            if self.isAugument:
                scale_ratios = np.linspace(-0.25, 0.25, NUM_OF_SCALES) + 1
                
                for ratio in scale_ratios:
                    if self.isPair:
                        p_midi_path = (self.performance_folder + row['midi_path']).replace(".mid", "_" + str(ratio) + ".mid")
                        s_midi_path = (self.score_folder + row['midi_path']).replace(".mid", "_" + str(ratio) + ".mid")
                        
                        p_tokens = self.tokenizer.tokenize_midi_file(p_midi_path, row['artist_id'])
                        
                        try:
                            s_tokens = self.tokenizer.tokenize_midi_file(s_midi_path, row['artist_id'])
                        except:
                            print("No score file: %s" % s_midi_path)
                            continue
                        
                        print(len(s_tokens))
                        s_tokens, p_tokens= self.align_p_and_s_tokens(s_tokens, p_tokens)
                        print(len(s_tokens))
                        
                        input_seqs = [p_tokens, s_tokens]
                    
                    else:
                        midi_path = (self.data_folder + row['midi_path']).replace(".mid", "_" + str(ratio) + ".mid")
                        tokens = self.tokenizer.tokenize_midi_file(midi_path, row['aritist_id'])
                        input_seqs = [tokens]
                        
                    self.create_segments_for_token_sequences(input_seqs)
            else:
                if self.isPair:
                    p_midi_path = self.performance_folder + row['midi_path']
                    s_midi_path = self.score_folder + row['midi_path']
                    
                    p_tokens = self.tokenizer.tokenize_midi_file(p_midi_path, row['artist_id'])
                    try:
                        s_tokens = self.tokenizer.tokenize_midi_file(s_midi_path, row['artist_id'])
                    except:
                        print("No score file: %s" % s_midi_path)
                        continue
                    
                    print(len(s_tokens))
                    s_tokens, p_tokens = self.align_p_and_s_tokens(s_tokens, p_tokens)
                    print(len(s_tokens))
                    
                    input_seqs = [p_tokens, s_tokens]
                
                else:
                    midi_path = self.data_folder + row['midi_path']
                    tokens = self.tokenizer.tokenize_midi_file(midi_path, row['aritist_id'])
                    input_seqs = [tokens]
                
                self.create_segments_for_token_sequences(input_seqs)
                
        # add to list
        self.x_list = np.asarray(self.x_list)
        self.y_list = np.asarray(self.y_list)
        self.mask_list = np.asarray(self.mask_list)
        self.performer_list = np.asarray(self.performer_list)
        self.piece_name_list = np.asarray(self.piece_name_list)
        
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=2)
        for train_index, val_index in sss.split(self.x_list, self.performer_list):
            train_x, valid_x = self.x_list[train_index], self.x_list[val_index]
            train_y, valid_y = self.y_list[train_index], self.y_list[val_index]   
            train_mask, valid_mask = self.mask_list[train_index], self.mask_list[val_index]
            train_performer, valid_performer = self.performer_list[train_index], self.performer_list[val_index]
            train_piece_name, valid_piece_name = self.piece_name_list[train_index], self.piece_name_list[val_index]
        
        print("Number of sequences in train: %d" % train_x.shape[0])
        print("Number of sequences in valid: %d" % valid_x.shape[0])
        
        np.savez(
            self.save_path,
            x_train=train_x,
            x_valid=valid_x,
            y_train=train_y,
            y_valid=valid_y,
            mask_train=train_mask,
            mask_valid=valid_mask,
            performer_train=train_performer,
            performer_valid=valid_performer,
            piece_name_train=train_piece_name,
            piece_name_valid=valid_piece_name
        )
        
        pprint.pprint(self.tokenizer.tokenizer.vocab)
        np.save(self.save_path_tokenizer, self.tokenizer.tokenizer)

class CreateRealValueDataset():
    """
    Create dataset of token values for training
    """
    def __init__(self, 
                 path_to_data_csv=str,
                 save_path_of_data=str,
                 data_folder_path=None,
                 max_len=MAX_LEN,
                 number_of_categories=6,
                 isPair=False,
                 isAugument=False):
        
        self.df = pd.read_csv(path_to_data_csv)
        
        if isPair:
            self.performance_folder = data_folder_path[0]
            self.score_folder = data_folder_path[1]
            
        else:
            self.data_folder = data_folder_path[0]
            
        self.isPair = isPair
        self.isAugument = isAugument
        self.max_len = max_len
        self.number_of_categories = number_of_categories
        self.save_path = save_path_of_data
            
        self.x_list = list()
        self.y_list = list()
        self.performer_list = list()
        self.mask_list = list()
        self.piece_name_list = list()

    @staticmethod
    def align_p_and_s_notes(ref_notes, target_notes):
        target_list = []
        extra = []
        for i in range(len(ref_notes)):
            if i >= len(target_notes) - 1:
                extra.append(i)
                continue
            if (ref_notes[i].pitch == target_notes[i].pitch) and \
            (ref_notes[i].velocity == target_notes[i].velocity):
                target_list.append(target_notes[i])
            else:
                is_find = False
                if i < 50: 
                    start = 0
                else:
                    start = i - 50
                for j in range(start, len(target_notes)):
                    if (ref_notes[i].pitch == target_notes[j].pitch) and \
                    (ref_notes[i].velocity == target_notes[j].velocity):
                        if len(target_list) > 0:
                            if np.abs(target_list[-1].start - target_notes[j].start) < 4:
                                target_list.append(target_notes[j])
                                is_find = True
                                break
                        else:
                            target_list.append(target_notes[j])
                            is_find = True
                            break
                if is_find == False:
                    extra.append(i)

        ref_list = [i for j, i in enumerate(ref_notes) if j not in extra]
        
        if len(ref_list) == len(target_list):
            return ref_list, target_list
        else:
            print("fail to align")
            raise ValueError    

    @staticmethod
    def load_midi_to_note_sequences(midi_path):
        midi = PrettyMIDI(midi_path)
        midi_notes = []
        notes = itertools.chain(*[
                inst.notes for inst in midi.instruments
                if inst.program in range(128) and not inst.is_drum])
        
        midi_notes += notes
        midi_notes.sort(key=lambda note: note.start)
        return midi_notes
    
    def _add_features(self, i, notes, feature_list):
        feature_list.append(notes[i].pitch)
        feature_list.append(notes[i].velocity)
        feature_list.append(notes[i].end - notes[i].start)
        feature_list.append(notes[i].start)
        if i < len(notes) - 1:
            feature_list.append(notes[i+1].start - notes[i].start)
            feature_list.append(notes[i+1].start - notes[i].end)
        else:
            feature_list.append(0)
            feature_list.append(0)
        return feature_list

    def create_segments_for_real_value_sequences(self, input_seqs):
        if self.isPair:
            x_list, mask_list = CreateTokenDataset.segment_sequences_with_attention_masks(input_seqs[1], 
                                                                        self.max_len, 
                                                                        self.number_of_categories)
            self.x_list += x_list
            self.mask_list += mask_list
            
            y_list, _ = CreateTokenDataset.segment_sequences_with_attention_masks(input_seqs[0], 
                                                                        self.max_len, 
                                                                        self.number_of_categories)
            self.y_list += y_list    
            assert len(self.x_list) == len(self.y_list)
            
        else:
            x_list, mask_list = CreateTokenDataset.segment_sequences_with_attention_masks(input_seqs[0], 
                                                                        self.max_len, 
                                                                        self.number_of_categories)
            self.x_list += x_list
            self.mask_list += mask_list
        
        performer_list = list()
        piece_name_list = list()
        for i in range(len(x_list)):
            performer_list.append(self.row['artist_id'])
            piece_name_list.append(self.row['midi_path'])
        
        self.performer_list += performer_list
        self.piece_name_list += piece_name_list
        
    def process(self):
        for idx, row in tqdm(self.df.iterrows(), 
                         total= self.df.shape[0], 
                         bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:20}{r_bar}'):
            self.row = row
    
            if self.isAugument:
                scale_ratios = np.linspace(-0.25, 0.25, 3) + 1
                for ratio in scale_ratios:
                    if self.isPair:
                        p_midi_path = (self.performance_folder + row['midi_path']).replace(".mid", "_" + str(ratio) + ".mid")
                        s_midi_path = (self.score_folder + row['midi_path']).replace(".mid", "_" + str(ratio) + ".mid")
                        
                        p_notes = self.load_midi_to_note_sequences(p_midi_path)
                        s_notes = self.load_midi_to_note_sequences(s_midi_path)
                        p_notes, s_notes = self.align_p_and_s_notes(p_notes, s_notes)

                        p_features = []
                        s_features = []
                        
                        for i in range(len(p_notes)):
                            p = []
                            s = []
                            p = self._add_features(i, p_notes, p)
                            s = self._add_features(i, s_notes, s)
                            p_features.append(p)
                            s_features.append(s)
                        
                        assert len(p_features) == len(s_features)
                        input_seqs = [p_features, s_features]
                    
                    else:
                        midi_path = (self.data_folder + row['midi_path']).replace(".mid", "_" + str(ratio) + ".mid")
                        notes = self.load_midi_to_note_sequences(midi_path)
                        features = []
                        for i in range(len(notes)):
                            tmp = []
                            tmp = self._add_features(i, notes, tmp)
                            features.append(tmp)                        
                        
                        input_seqs = [features]
                        
                    self.create_segments_for_real_value_sequences(input_seqs)
            else:
                if self.isPair:
                    p_midi_path = self.performance_folder + row['midi_path']
                    s_midi_path = self.score_folder + row['midi_path']
                    
                    p_notes = self.load_midi_to_note_sequences(p_midi_path)
                    s_notes = self.load_midi_to_note_sequences(s_midi_path)
                    p_notes, s_notes = self.align_p_and_s_notes(p_notes, s_notes)

                    p_features = []
                    s_features = []
                    
                    for i in range(len(p_notes)):
                        p = []
                        s = []
                        p = self._add_features(i, p_notes, p)
                        s = self._add_features(i, s_notes, s)
                        p_features.append(p)
                        s_features.append(s)
                    
                    input_seqs = [p_features, s_features]
                
                else:
                    midi_path = self.data_folder + row['midi_path']
                    notes = self.load_midi_to_note_sequences(midi_path)
                    features = []
                    for i in range(len(notes)):
                        tmp = []
                        tmp = self._add_features(i, notes, tmp)
                        features.append(tmp)                        
                    
                    input_seqs = [features]
                    
                self.create_segments_for_real_value_sequences(input_seqs)
                
        
        # add to list
        self.x_list = np.asarray(self.x_list)
        self.y_list = np.asarray(self.y_list)
        self.mask_list = np.asarray(self.mask_list)
        self.performer_list = np.asarray(self.performer_list)
        self.piece_name_list = np.asarray(self.piece_name_list)
        
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=2)
        for train_index, val_index in sss.split(self.x_list, self.performer_list):
            train_x, valid_x = self.x_list[train_index], self.x_list[val_index]
            train_y, valid_y = self.y_list[train_index], self.y_list[val_index]   
            train_mask, valid_mask = self.mask_list[train_index], self.mask_list[val_index]
            train_performer, valid_performer = self.performer_list[train_index], self.performer_list[val_index]
            train_piece_name, valid_piece_name = self.piece_name_list[train_index], self.piece_name_list[val_index]
        
        print("Number of sequences in train: %d" % train_x.shape[0])
        print("Number of sequences in valid: %d" % valid_x.shape[0])
        
        np.savez(
            self.save_path,
            x_train=train_x,
            x_valid=valid_x,
            y_train=train_y,
            y_valid=valid_y,
            mask_train=train_mask,
            mask_valid=valid_mask,
            performer_train=train_performer,
            performer_valid=valid_performer,
            piece_name_train=train_piece_name,
            piece_name_valid=valid_piece_name
        )
        
if __name__ == "__main__":
    if sys.argv[1] == "token":
        tokenizer = Tokenizer(path_to_tokenizer="tokenizers/mytokenizer_s2p.npy")
        data_processor = CreateTokenDataset("/import/c4dm-04/jt004/ATEPP-data-exp/ATEPP-metadata-s2p_align.csv",
                                            "data/s2p_data",
                                            "tokenizers/mytokenizer_s2p.npy",
                                            data_folder_path=[
                                                "/import/c4dm-04/jt004/ATEPP-data-exp/performance_aligned/",
                                                "/import/c4dm-04/jt004/ATEPP-data-exp/score_aligned/"
                                            ],
                                            tokenizer=tokenizer,
                                            isPair=True,
                                            isAugument=False)
    
    elif sys.argv[1] == "real":
        data_processor = CreateRealValueDataset("/import/c4dm-04/jt004/ATEPP-data-exp/ATEPP-metadata-s2p_align.csv",
                                                "data/s2p_data_real",
                                                data_folder_path=[
                                                "/import/c4dm-04/jt004/ATEPP-data-exp/performance_aligned/",
                                                "/import/c4dm-04/jt004/ATEPP-data-exp/score_aligned/"
                                                ],
                                                isPair=True,
                                                isAugument=False)
    
    data_processor.process()    