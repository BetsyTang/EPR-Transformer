PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
MASK_ID = 3
NUM_PERFORMER = 6

TOKENIZER_PARAMS = {
    "pitch_range": (21, 109),
    "beat_res": {(0, 12): 384},
    "num_velocities": 64,
    "special_tokens": ["PAD", "BOS", "EOS", "MASK"],
    "use_chords": False,
    "use_rests": False,
    "use_tempos": False,
    "use_time_signatures": False,
    "use_programs": False,
    "num_tempos": 32,  # number of tempo bins
    "tempo_range": (40, 250),  # (min, max)
    "data_type": "Performance",
    "remove_outliers": True
}
