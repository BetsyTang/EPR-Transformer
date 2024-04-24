# @Script: data_preprocessing.py
# @Last Modified By: Jingjing Tang
# @Last Modified At: 2024-04-18 17:35:45ints, num_of_features) to max_len with PAD_ID and create attention mask.

from data_config import *

TICKS_PER_BEAT = 384
BEAT_PER_BAR = 4
PITCH_TOKEN = 0
VEL_TOKEN = 1
BAR_TOKEN = 4

def pad_sequence_with_attention(sequence, max_len, padding=True):
    r"""
    Args:
    - sequence: A list of lists representing the sequence.
    - max_len: The maximum length to pad the sequence to.

    Returns:
    - padded_sequences: List of padded sequences.
    - attention_masks: List of attention masks corresponding to padded sequences.
    """
    sub_sequences = []
    padded_sequences = []
    attention_masks = []
    for i in range(0, len(sequence), max_len):
        sub_sequence = sequence[i:i+max_len]
        if isinstance(sub_sequence[0], list) == False:
            sub_sequence = [sub_sequence]
        sub_sequences.append(sub_sequence)
        if padding == True:
            sub_padded_sequence, sub_attention_mask = _pad_sub_sequence_with_mask(sub_sequence, max_len)
            padded_sequences.append(sub_padded_sequence)
            attention_masks.append(sub_attention_mask)
            assert len(padded_sequences) == len(attention_masks)
    
    return (padded_sequences, attention_masks) if padding else (sub_sequences, None)

def _pad_sub_sequence_with_mask(sub_sequence, max_len):
    """
    Pad a sub-sequence to max_len with PAD_ID and create attention mask.

    Args:
    - sub_sequence: A list representing a sub-sequence.
    - max_len: The maximum length to pad the sub-sequence to.

    Returns:
    - padded_sub_sequence: The padded sub-sequence.
    - sub_mask: The attention mask for the padded sub-sequence.
    """
    sub_mask = [1] * len(sub_sequence) + [0] * (max_len - len(sub_sequence))
    if len(sub_sequence) >= max_len:
        return sub_sequence[:max_len], sub_mask[:max_len]
    else:
        padded_sub_sequence = sub_sequence + [[PAD_ID] * len(sub_sequence[0])] * (max_len - len(sub_sequence))
        return padded_sub_sequence, sub_mask

def align_performance_and_score(ref_data, target_data, TICKS_PER_BEAT):
    """
    Aligns reference data (either tokens or MIDI notes) with target data.
    
    Args:
        ref_data (list): List of reference tokens or MIDI notes.
        target_data (list): List of target tokens or MIDI notes.
        TICKS_PER_BEAT (int): Ticks per beat used for timing calculations.
    
    Returns:
        tuple: A tuple containing two lists: aligned reference data and aligned target data.
    
    Raises:
        ValueError: If the alignment process fails.
    """
    if isinstance(ref_data[0], dict):  # Assuming token data is dictionary-based
        return align_tokens(ref_data, target_data, TICKS_PER_BEAT)
    elif hasattr(ref_data[0], 'pitch'):  # Assuming MIDI note-like objects with attributes
        return align_midi_notes(ref_data, target_data)
    else:
        raise ValueError("Unsupported data type for alignment")

def align_tokens(ref_tokens, target_tokens, TICKS_PER_BEAT):
    """
    Align transcribed scores with the transcribed performance based on tokens.
    """
    target_list = []
    unmatched_indices = []

    for i, ref_token in enumerate(ref_tokens):
        if i >= len(target_tokens):
            unmatched_indices.append(i)
            continue

        start = max(0, i - 10)
        end = min(len(target_tokens), i + 20)

        match = find_matching_token(ref_token, target_tokens, start, end, target_list[-1] if target_list else None, TICKS_PER_BEAT)
        if match:
            target_list.append(match)
        else:
            unmatched_indices.append(i)

    ref_list = [ref_tokens[i] for i in range(len(ref_tokens)) if i not in unmatched_indices]

    if len(ref_list) == len(target_list):
        return ref_list, target_list
    else:
        print("Fail to align two sequences")
        raise ValueError("Alignment failed due to unmatched tokens.")

def align_midi_notes(ref_notes, target_notes):
    """
    Align MIDI notes from a reference performance and a target score.
    """
    target_list = []
    extra_indices = []

    for i, ref_note in enumerate(ref_notes):
        if i >= len(target_notes):
            extra_indices.append(i)
            continue

        if ref_note.pitch == target_notes[i].pitch and ref_note.velocity == target_notes[i].velocity:
            target_list.append(target_notes[i])
        else:
            found_match = False
            start = 0 if i < 50 else i - 50

            for j in range(start, len(target_notes)):
                if ref_note.pitch == target_notes[j].pitch and ref_note.velocity == target_notes[j].velocity:
                    if target_list and abs(target_list[-1].start - target_notes[j].start) < 4:
                        target_list.append(target_notes[j])
                        found_match = True
                        break
                    elif not target_list:
                        target_list.append(target_notes[j])
                        found_match = True
                        break

            if not found_match:
                extra_indices.append(i)

    ref_list = [ref_note for j, ref_note in enumerate(ref_notes) if j not in extra_indices]
    
    if len(ref_list) == len(target_list):
        return ref_list, target_list
    else:
        print("Fail to align")
        raise ValueError("Alignment failed due to unmatched notes.")
    
def is_within_time_bounds(target_time, last_target_time, TICKS_PER_BEAT):
    """Check if the time difference between reference and target times is within allowed bounds."""
    if last_target_time is not None:
        time_difference = target_time * BEAT_PER_BAR * TICKS_PER_BEAT + target_time - last_target_time * BEAT_PER_BAR * TICKS_PER_BEAT - last_target_time
        return -BEAT_PER_BAR * TICKS_PER_BEAT < time_difference
    return True

def find_matching_token(ref_token, target_tokens, start, end, last_target_token, TICKS_PER_BEAT):
    """Attempt to find a matching token in the target tokens within the specified range."""
    for j in range(start, end):
        target_token = target_tokens[j]
        if (ref_token[PITCH_TOKEN] == target_token[PITCH_TOKEN]) and (ref_token[VEL_TOKEN] == target_token[VEL_TOKEN]):
            if last_target_token is None or is_within_time_bounds(target_token[BAR_TOKEN], last_target_token[BAR_TOKEN], TICKS_PER_BEAT):
                return target_token
    return None