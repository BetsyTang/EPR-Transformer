# Copyright (c) 2024 Jingjing Tang
#
# -*- coding:utf-8 -*-
# @Script: expression_dataset.py
# @Author: Jingjing Tang
# @Email: tangjingjingbetsy@gmail.com
# @Create At: 2024-04-15 17:02:52
# @Last Modified By: Jingjing Tang
# @Last Modified At: 2024-04-24 19:24:21
# @Description: This is data_processing for transcribed scores.

from expression_tokenizer import *
from tools import pad_sequence_with_attention
from config import *

from torch.utils.data import Dataset
from tqdm import tqdm
import argparse
import os
import itertools
import logging
import datetime
import time
import torch

class MidiLoader():
    def __init__(self, alignment):
        self.alignment = alignment
    
    def __call__(self, filepath):
        if self.alignment:
            note_seqs = self.load_alignments(filepath)
        else:
            note_seqs = self.load_midi(filepath)
            
        return self.add_features(note_seqs)
       
    def load_midi(self, midi_path):
        midi = Score(midi_path, ttype="second")
        midi_notes = []
        notes = itertools.chain(*[
                inst.notes for inst in midi.tracks
                if inst.program in range(128) and not inst.is_drum])
        midi_notes += notes
        midi_notes.sort()
        return [midi_notes]
    
    def load_alignments(self, align_file):
        align_file = ExpressionTok.load_alignments(align_file)
        perf_notes = []
        score_notes = []
        for idx, row in align_file.iterrows():
            if row['alignOfftime'] == row['alignOntime']:
                row['alignOfftime'] += 0.5
            perf_notes.append(Note(row['alignOntime'], 
                                   row['alignOfftime'] - row['alignOntime'],
                                   row['alignPitch'],
                                   row['alignOnvel'],
                                   ttype='second'))
            
            if row['refOfftime'] == row['refOntime']:
                row['refOfftime'] += 0.5
            score_notes.append(Note(row['refOntime'], 
                                   row['refOfftime'] - row['refOntime'],
                                   row['refPitch'],
                                   60,
                                   ttype='second'))
            
        assert len(perf_notes) == len(score_notes)
        return [perf_notes, score_notes]
    
    @staticmethod
    def calcuate_ioi(notes):
        ioi = [0 if i == 0 else notes[i].start - notes[i-1].start for i in range(len(notes))]
        return ioi
    
    @staticmethod
    def calcuate_durdev(perf_notes, score_notes):
        assert len(perf_notes) == len(score_notes)
        durdev = [perf_notes[i].duration - score_notes[i].duration for i in range(len(perf_notes))]
        return durdev
    
    def _add_features(self, notes):
        ioi = MidiLoader.calcuate_ioi(notes)
        features = [[note.pitch for note in notes],
                    [note.velocity for note in notes],
                    [note.duration for note in notes],
                    ioi,
                    [note.start for note in notes]
                    ]
        return features
    
    def add_features(self, note_seqs):
        if len(note_seqs) > 1:
            features = self._add_features(note_seqs[0])
            features += self._add_features(note_seqs[1])
            # durdev = MidiLoader.calcuate_durdev(note_seqs[0], note_seqs[1])
            # features += [durdev]
        else:
            features = self._add_features(note_seqs[0])
        
        return np.array(features).T.tolist() 
      
class TokenLoader():
    """
    Self-defined Tokenizer Object for easy usage base on MidTok Tokenizer
    """
    def __init__(self, args):
        
        self.alignment = args.alignment
        self.config = TokenizerConfig(**args.tokenizer_config)
        if args.load_tokenizer != None:
            self.tokenizer = ExpressionTok(params=Path(args.load_tokenizer))
        else:
            self.tokenizer = ExpressionTok(self.config)
        
        self.output_tokenizer = args.output_tokenizer
    
    def __call__(self, filepath, midipath=None):
        if self.alignment:
            midi = Score(midipath)
            tokens = self.tokenizer.alignment_to_token(filepath, midi)
        else:
            tokens = self.tokenizer(Path("to", filepath))[0] # The default midi_to_tokens return a list[TokSequence]
        return tokens.ids

    def token2midi(self, tokens, output_paths):
        r"""transfer the tokens to a midi file

        Args:
            tokens (list): list of tokens
            output_paths (list[strs]): path to the save directionary, output_paths[0] for performance,
                                       output_paths[1] for scores.
        """
        if self.config.additional_params["data_type"] == "Alignment":
            self. tokenizer.align_tokens_to_midi(tokens, output_paths[0], output_paths[1])
        else:
            self.tokenizer.tokens_to_midi([tokens], output_path=output_paths[0])
    
    def save(self):
        self.tokenizer.save_params(self.output_tokenizer)

class ExpressionDataset():
    """
    Create dataset of token values for training
    """
    def __init__(self, args, logger=None):
        print("Initialzing....")
        self.df = pd.read_csv(args.csv_file)
        self.data_folders = args.data_folders
        
        if args.mode == "token":
            self.loader = TokenLoader(args)
            self.save_tokenizer = args.output_tokenizer
            self.bos_id = BOS_ID
            self.eos_id = EOS_ID
        else:
            self.loader = MidiLoader(args.alignment)
            self.bos_id = -1
            self.eos_id = -2
            
        self.mode = args.mode
                
        self.alignment = args.alignment
        self.transcribe = args.transcribe
        self.score = args.score
        self.padding = args.padding
        self.split = args.split
        self.compact = args.compact
        
        self.max_len = args.max_len
        self.save_data = args.output_data
            
        self.data = {'performance':dict(), 
                     'score':dict()}
        
        self.logger = logger #Not Using
        
    def __call__(self, csv_file=None):
        start_time = time.time()
        if csv_file != None:
            self.df = pd.read_csv(csv_file)
            
        idxs = self.df.index.tolist()
        print("Start Processing ...")
        for idx in tqdm(idxs):
            meta = self.load_metadata(idx)
            if self.alignment:
                sequence = self.loader(meta['align_file'], meta['perf_midi']) if self.mode == 'token' else self.loader(meta['align_file'])
                seqs, masks = self.create_subsequences(sequence) 
                perf_seqs = []
                score_seqs = []
                for seq in seqs:
                    if self.mode == "token":
                        perf_seqs.append([s[0:6] for s in seq])
                        score_seqs.append([[s[0]] + s[6:-1] for s in seq]) #Add Pitch to score sequences
                    else:
                        try:
                            perf_seqs.append([s[0:5] for s in seq])
                            score_seqs.append([s[5:-1] for s in seq])
                        except:
                            print(seq)
                            raise ValueError
                
                self.save_subsequences(meta, perf_seqs, masks, "performance")
                self.save_subsequences(meta, score_seqs, masks, "score")
            else:
                perf = self.loader(meta['perf_midi'])
                seqs, masks = self.create_subsequences(perf)         
                self.save_subsequences(meta, seqs, masks, "performance")
                if self.score:
                    score = self.loader(meta['score_midi'])
                    seqs, masks = self.create_subsequences(score)
                    self.save_subsequences(meta, seqs, masks, "score")
        
        print(f"Performances size: {len(self.data['performance'])}")
        print(f"Scores size: {len(self.data['score'])}")
        
        if self.split:  
            print("Spliting Dataset & Saving...")          
            self.split_dataset()
        else:
            if self.save_data:
                print("Saving...")    
                if self.compact:      
                    np.savez(
                        self.save_data,
                        performance = self.data['performance'],
                        score = self.data['score']
                    )
                else:
                    for data_type in ['performance', 'score']:
                        for idx in self.data[data_type]:
                            save_dir = os.path.join(os.path.dir(self.save_data), data_type)
                            if os.path.isdir(save_dir) == False:
                                os.makedirs(save_dir)
                            np.save(os.path.join(save_dir, idx, ".npy"), self.data[data_type][idx])
            
        if self.mode == "token":
            self.loader.save()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        time_delta = datetime.timedelta(seconds=elapsed_time)
        print(f"Running Time: {time_delta}")
        return self.data
    
    def split_dataset(self):
        perfs = self.data['performance']
        def add_item_to_split(split):
            if self.score or self.alignment:
                scores = self.data['score']
                return [{'perf_id': i, 
                        'perf_seq': perfs[i]['seq'], 
                        'score_seq': scores[i]['seq'], 
                        'perf_mask': perfs[i]['mask'], 
                        'performer_id': perfs[i]['performer_id']} \
                        for i in perfs.keys() if perfs[i]['split'] == split]
            else:
                return [{'perf_id': i, 
                        'perf_seq': perfs[i]['seq'], 
                        'perf_mask': perfs[i]['mask'], 
                        'performer_id': perfs[i]['performer_id']} \
                        for i in perfs.keys() if perfs[i]['split'] == split]
        
        train_set = add_item_to_split('train')
        validation_set = add_item_to_split('validation')
        test_set = add_item_to_split('test')
        
        print(f"Train size: {len(train_set)}")
        print(f"Validation size: {len(validation_set)}")
        print(f"Test size: {len(test_set)}")
        
        if self.save_data:
            if self.compact:
                np.savez(
                    self.save_data,
                    performance = self.data['performance'],
                    score = self.data['score'],
                    train = train_set,
                    validation = validation_set,
                    test = test_set
                )
            else:
                for split, data_set in enumerate([('train', train_set), 
                                                ('validation', validation_set), 
                                                ('test', test_set)]):
                    for data in data_set:
                        save_dir = os.path.join(os.path.dir(self.save_data), split)
                        if os.path.isdir(save_dir) == False:
                            os.makedirs(save_dir)
                        np.save(os.path.join(save_dir, data['perf_id'], ".npy"), data)
        else:
            self.data['train_set'] = train_set
            self.data['validation_set'] = validation_set
            self.data['test_set'] = test_set
        
    def save_subsequences(self, meta, seqs, masks, mtype):
        path = meta['score_midi'] if mtype == 'score' else meta['perf_midi']
        # The following dictionary could be modified according to different demands
        for i in range(len(seqs)):
            self.data[mtype][f"{meta['perf_id']}_{i}"] = \
                {
                    'seq':seqs[i],
                    'mask':masks[i],
                    'performer':meta['performer'],
                    'performer_id':meta['performer_id'],
                    'midi_path':path,
                    'composition':meta['composition'],
                    'split':meta['split']
                }
    
    def create_subsequences(self, sequence):
        # print(sequence[0])
        if sequence and isinstance(sequence[0], list):
            category_length = len(sequence[0])

            # Create a list with 'BOS_ID' repeated 'category_length' times, wrapped in 
            # a list to match the structure of 'sequence'
            bos_row = [[self.bos_id] * category_length]

            # Create a list with 'EOS_ID' repeated 'category_length' times, similarly wrapped
            eos_row = [[self.eos_id] * category_length]

            # Concatenate bos_row, sequence, and eos_row
            sequence = bos_row + sequence + eos_row
        else:
            print("Error: sequence is empty or not properly structured.")
            raise ValueError
        
        sequences, masks = pad_sequence_with_attention(sequence, self.max_len, self.padding)
        
        return sequences, masks
    
    def load_metadata(self, idx):
        # The following dictionary could be modified according to different demands
        meta = {}
        meta['performer'] = self.df.loc[idx, 'artist']
        meta['performer_id'] = self.df.loc[idx, 'artist_id']
        meta['perf_id'] = str(self.df.loc[idx, 'perf_id']).zfill(5)
        meta['album'] = self.df.loc[idx, 'album_id']
        meta['perf_midi'] = os.path.join(self.data_folders[0], self.df.loc[idx, 'midi_path'])
        meta['composition'] = self.df.loc[idx, 'composition_id']
        meta['split'] = self.df.loc[idx, 'split']
        if self.transcribe:
            meta['score_midi'] = os.path.join(self.data_folders[1], self.df.loc[idx, 'midi_path'])
        else:
            meta['score_midi'] = os.path.join(self.data_folders[0], self.df.loc[idx, 'score_path'] + ".midi")
        if self.alignment:
            meta['align_file'] = os.path.join(self.data_folders[0], self.df.loc[idx, 'align_file'])
        return meta

class DealDataset(Dataset):
    def __init__(self, x_data, mask_data, y_data, style, idx, alignment=True):
        self.x_data = torch.FloatTensor(x_data)
        self.mask_data = torch.FloatTensor(mask_data)
        self.y_data = torch.FloatTensor(y_data)
        self.style = torch.LongTensor(style)
        self.idx = idx
        self.len = self.x_data.shape[0]
        self.alignment = alignment
        
        # x_features = ["Pitch", "Velocity", "Duration", "IOI", "Position", "Bar"]
        y_features = ["Velocity", "IOI", "Duration"]
        self.y_data = self.y_data[:, :, self.get_feature_index(y_features)] 
    
    def __getitem__(self, index):
        return self.x_data[index], self.mask_data[index], self.y_data[index],  self.style[index], self.idx[index]

    def __len__(self):
        return self.len

    def get_feature_index(self, feature_names, mtype="perf"):
        feature_list = ["Pitch", "Velocity", "Duration", "IOI", "Position", "Bar"]        
        feature_idx = [feature_list.index(name) for name in feature_names]
        
        if len(feature_names) == 0:
            return feature_idx[0]
        else:
            return feature_idx 
            
# Set up logging to file
def set_logger(logger_name, logeer_file):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(logeer_file)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
 
def get_args():
    parser = argparse.ArgumentParser(description='A simple program to demonstrate argparse')
    parser.add_argument('-c', '--csv_file', default=None, help='Path to the CSV file containing metadata. Defaults to "CSV_FILE".')
    parser.add_argument('-o', '--output_data', type=str, default='data/data.npz', help='Path where the processed data will be stored. Defaults to "data/data.npz".')
    parser.add_argument('-t', '--output_tokenizer', type=str, default='data/tokenizer.json', help='Path where the tokenizer configuration will be saved. Defaults to "data/tokenizer.json".')
    parser.add_argument('-l', '--load_tokenizer', default=None, help='Optional path to a previously saved tokenizer to be loaded.')
    parser.add_argument('-g', '--tokenizer_config', type=dict, default=TOKENIZER_PARAMS, help='Tokenizer configuration as a dictionary. Defaults to "TOKENIZER_PARAMS".')
    parser.add_argument('-d', '--data_folders', nargs="+", help='List of paths to data folders. Multiple folders can be specified. Defaults to "PATH_TO_DATA".')
    parser.add_argument('-A', '--alignment', action='store_true', help='Enable use of alignment data. Disabled by default.')
    parser.add_argument('-S', '--score', action='store_true', help='Enable the use of musical score data. Disabled by default.')
    parser.add_argument('-s', '--split', action='store_true', help='Enable to split the dataset into train, validation, test. Disabled by default.')
    parser.add_argument('-T', '--transcribe', action='store_true', help='Enable the use of transcribed score data. Disabled by default.')
    parser.add_argument('-P', '--padding', action='store_false', help='Do NOT pad shorter sequences to the max_len.')
    parser.add_argument('-C', '--compact', action='store_true', help='To save the data in the compact file, not in different files')
    parser.add_argument('-m', '--mode', choices=['real', 'token'], default='token', help='Set the data format for dataset creation. Choices are "real" or "token", with "token" as the default.')
    parser.add_argument('-ln', '--logger_name', type=str, default='run', help='Name for the logger. Used to tag log entries. Defaults to "run".')
    parser.add_argument('-ml', '--max_len', type=int, default=1000, help='Maximum length of the sequences to be processed. Defaults to 1000.')
    parser.add_argument('-lf', '--logger_file', type=str, default='logs/run.log', help='File path where logs will be written. Defaults to "logs/run.log".')
    args = parser.parse_args()
    
    parser.print_help()
    
    return args

def print_arg_values(args):
    print("################# Argument Values ######################")
    print(f"CSV File Path: {args.csv_file}")
    print(f"Output Data Path: {args.output_data}")
    print(f"Output Tokenizer Path: {args.output_tokenizer}")
    print(f"Load Tokenizer Path: {args.load_tokenizer}")
    print(f"Tokenizer Configuration: {args.tokenizer_config}")
    print(f"Data Folders: {args.data_folders}")
    print(f"Alignment Enabled: {args.alignment}")
    print(f"Score Data Enabled: {args.score}")
    print(f"Dataset Split Enabled: {args.split}")
    print(f"Transcribed Score Data Usage: {args.transcribe}")
    print(f"Not Padding Short Sequences: {'Yes' if args.padding else 'No'}")
    print(f"Save in a Compact File: {'Yes' if args.compact else 'No'}")
    print(f"Data Mode: {args.mode}")
    print(f"Logger Name: {args.logger_name}")
    print(f"Maximum Length of Sequences: {args.max_len}")
    print(f"Logger File Path: {args.logger_file}")
    print("########################################################")
    now = datetime.datetime.now()
    print("Current Date and Time:", now.strftime("%Y-%m-%d %H:%M:%S"))
        
if __name__ == "__main__":
    
    args = get_args()
    
    if args.alignment:
        args.tokenizer_config['data_type'] = "Alignment"
    else:
        args.tokenizer_config['data_type'] = "Midi"
        
    print_arg_values(args)
    
    # logger = set_logger(args.logger_name, args.logger_file)
    
    creater = ExpressionDataset(args)
    creater()