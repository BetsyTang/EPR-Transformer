# [Branch Test] RHEPP-Transformer: Reconstructing Human Expressiveness in Piano Performances with a Transformer Network

We updated the implementation for the paper [Reconstructing Human Expressiveness in Piano Performances with a Transformer Network](https://arxiv.org/abs/2306.06040).

## Data Preporcessing
We provide scripts to convert a dataset of midi files or alignments (`*_infer_corresp.txt` obtained from [Nakamura's algorithm](https://midialignment.github.io/demo.html)) to note sequences. The resulted sequences are in shape `number_of_sequences_in_total x number_of_notes_per_sequence x number_of_features_used`, or simply `(segments, notes, features)`. We offer options to generate either token sequences (using the `ExpressionTokenizer` developed with the `MidiTok` library) or real value sequences, meaning we use the values directly obtained from the midi files without tokenisation.  

The current scripts were assumed to be applied to the ATEPP dataset, and therefore, require a csv file containing meta information including midi file paths, performance ids, performer ids and so on. If you would use this script for your own dataset, please ensure you have all the information required, or you could rewrite functions `load_metadata()` (maybe also `save_subsequences()` and `split_dataset()`) for the `ExpressionDataset` class.

We provide an example, `data/data.sh`, to show how you could use the script to generate your own dataset with `data/expression_dataset.py`.

```bash
$./data/data.sh
```

A full list of arguments of how you could apply the scripts to create different kinds of datasets could be found by

```bash
python data/expression_dataset.py -h

### Full list of arguments
-c CSV_FILE, --csv_file CSV_FILE
                      Path to the CSV file containing metadata. Defaults to
                      "CSV_FILE".
-o OUTPUT_DATA, --output_data OUTPUT_DATA
                      Path where the processed data will be stored. Defaults
                      to "data/data_files/data.npz".
-t OUTPUT_TOKENIZER, --output_tokenizer OUTPUT_TOKENIZER
                      Path where the tokenizer configuration will be saved.
                      Defaults to "data/tokenizer.json".
-l LOAD_TOKENIZER, --load_tokenizer LOAD_TOKENIZER
                      Optional path to a previously saved tokenizer to be
                      loaded.
-g TOKENIZER_CONFIG, --tokenizer_config TOKENIZER_CONFIG
                      Tokenizer configuration as a dictionary. Defaults to
                      "TOKENIZER_PARAMS".
-f FEATURE_LIST, --feature_list FEATURE_LIST
                      Features to be used for midi. Defaults to "FEATURE
                      NAMES".
-d DATA_FOLDERS [DATA_FOLDERS ...], --data_folders DATA_FOLDERS [DATA_FOLDERS ...]
                      List of paths to data folders. Multiple folders can be
                      specified. Defaults to "PATH_TO_DATA".
-A, --alignment       Enable use of alignment data. Disabled by default.
-S, --score           Enable the use of musical score data. Disabled by
                      default.
-s, --split           Enable to split the dataset into train, validation,
                      test. Disabled by default.
-T, --transcribe      Enable the use of transcribed score data. Disabled by
                      default.
-P, --padding         Do NOT pad shorter sequences to the max_len.
-C, --compact         To save the data in the compact file, not in different
                      files
-m {real,token}, --mode {real,token}
                      Set the data format for dataset creation. Choices are
                      "real" or "token", with "token" as the default.
-ln LOGGER_NAME, --logger_name LOGGER_NAME
                      Name for the logger. Used to tag log entries. Defaults
                      to "run".
-ml MAX_LEN, --max_len MAX_LEN
                      Maximum length of the sequences to be processed.
                      Defaults to 1000.
-lf LOGGER_FILE, --logger_file LOGGER_FILE
                      File path where logs will be written. Defaults to
                      "logs/run.log".
```

Shell script usage is recommended, but you could also try command line usage if you prefer.

```bash
# To create a compact file `data.npz` with two subcategories `performance` and `score` (will be empty if no score file is used) to store the corresponding sequences. 
python data/expression_dataset.py -C

# To split the dataset (using -s) into train, validatoin, test set and save all the subsets to a compact file (-C). It will save a npz file with subcategories `performance`, `score`,  `train`, `validation`, and `test`.
python data/expression_dataset.py -s -C

# To process alignment files (-A)
python data/expression_dataset.py -A

# To set paths: output data file, output tokenizer file, path to the database folder
python data/expression_dataset.py -o data.npz -t tokenizer.json -d ['data/performances/',]
```

## Citation
```
@article{tang2023reconstructing,
  title={Reconstructing Human Expressiveness in Piano Performances with a Transformer Network},
  author={Tang, Jingjing and Wiggins, Geraint and Fazekas, George},
  journal={arXiv preprint arXiv:2306.06040},
  year={2023}
}
```
## Contact
Jingjing Tang: jingjing.tang@qmul.ac.uk


