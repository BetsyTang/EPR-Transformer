from __future__ import annotations

from symusic import Note, Score, Tempo, TimeSignature, Track
from miditok.classes import Event, TokSequence
from miditok.constants import MIDI_INSTRUMENTS, TIME_SIGNATURE, TEMPO
from miditok.midi_tokenizer import MIDITokenizer
from miditok.utils import (
    compute_ticks_per_bar, 
    compute_ticks_per_beat, 
    get_bars_ticks, 
    get_midi_ticks_per_beat, 
    detect_chords)
from miditok import TokenizerConfig
from pathlib import Path
import pandas as pd
import numpy as np
import os

TICKS_PER_BEAT = 384

class ExpressionTok(MIDITokenizer):
    r"""
    Expression tokenizer.

    * 0: Pitch;
    * 1: Performance Velocity; -> PVelocity
    * 2: Performance Duration; -> PDuration
    * 3: Performance Inter Onset Interval (Onset time difference between the current note and the previous note); -> PIOI
    * 4: Perfromance Position; -> PPosition
    * 5: Perfromance Bar; -> PBar
    * 6: Score Duration; -> SDuration
    * 7: Score Inter Onset Interval; -> SIOI
    * 8: Score Position; -> SPosition
    * 9: Score Bar; -> SBar
    * 10: Duration Deviation; -> SPDurationDev

    **Notes:**
    * Tokens are first sorted by time, then track, then pitch values.
    
    """
    
    # def __init__(self, tokenizer_config: TokenizerConfig = None, params: str | Path | None = None,):
    #     super().__init__(tokenizer_config, params)

    def _tweak_config_before_creating_voc(self) -> None:
        self.config.use_chords = False
        self.config.use_rests = False
        self.config.use_pitch_bends = False
        self.config.use_pitch_intervals = False
        self.config.use_tempos = False
        self.config.use_programs = False
        self.config.delete_equal_successive_tempo_changes = True
        self.config.program_changes = False
        self.config.use_time_signatures = False

        # used in place of positional encoding
        # This attribute might increase if the tokenizer encounter longer MIDIs
        if "max_bar_embedding" not in self.config.additional_params:
            self.config.additional_params["max_bar_embedding"] = 3000

        assert self.config.additional_params["data_type"] in ['Performance', 'Score', 'Alignment']
        
        if self.config.additional_params["data_type"] == "Performance":
            token_types = ["Pitch", "PVelocity", "PDuration", "PIOI", "PPosition", "PBar"]
        if self.config.additional_params["data_type"] == "Score":
            token_types = ["Pitch", "SDuration", "SIOI", "SPosition", "SBar"]
        if self.config.additional_params["data_type"] == "Alignment":
            token_types = ["Pitch", "PVelocity", "PDuration", "PIOI", "PPosition", "PBar", \
                            "SDuration", "SIOI", "SPosition", "SBar", "SPDurationDev"]
        
        self.vocab_types_idx = {
            type_: idx for idx, type_ in enumerate(token_types)
        }  # used for data augmentation

    def _add_time_events(
        self, events: list[Event], time_division: int
    ) -> list[list[Event]]:
        r"""
        Create the time events from a list of global and track events.

        Internal method intended to be implemented by child classes.
        The returned sequence is the final token sequence ready to be converted to ids
        to be fed to a model.

        :param events: sequence of global and track events to create tokens time from.
        :param time_division: time division in ticks per quarter of the MIDI being
            tokenized.
        :return: the same events, with time events inserted.
        """
        # Add time events
        all_events = []
        current_bar = 0
        current_bar_from_ts_time = 0
        current_tick_from_ts_time = 0
        current_pos = 0
        
        prev_bar = -1
        prev_pos = 0
        
        previous_tick = 0
        current_time_sig = TIME_SIGNATURE
        current_tempo = self.default_tempo
        current_program = None
        ticks_per_bar = compute_ticks_per_bar(
            TimeSignature(0, *current_time_sig), time_division
        )
        ticks_per_beat = compute_ticks_per_beat(current_time_sig[1], time_division)
        ticks_per_pos = ticks_per_beat // self.config.max_num_pos_per_beat
        pos_per_bar = max(self.config.beat_res.values())
        for e, event in enumerate(events):
            # Set current bar and position
            # This is done first, as we need to compute these values with the current
            # ticks_per_bar, which might change if the current event is a TimeSig
            if event.time != previous_tick:
                elapsed_tick = event.time - current_tick_from_ts_time
                current_bar = current_bar_from_ts_time + elapsed_tick // ticks_per_bar
                tick_at_current_bar = (
                    current_tick_from_ts_time
                    + (current_bar - current_bar_from_ts_time) * ticks_per_bar
                )
                current_pos = (event.time - tick_at_current_bar) // ticks_per_pos
                previous_tick = event.time

            if event.type_ == "TimeSig":
                current_time_sig = list(map(int, event.value.split("/")))
                current_bar_from_ts_time = current_bar
                current_tick_from_ts_time = previous_tick
                ticks_per_bar = compute_ticks_per_bar(
                    TimeSignature(event.time, *current_time_sig), time_division
                )
                ticks_per_beat = compute_ticks_per_beat(
                    current_time_sig[1], time_division
                )
                ticks_per_pos = ticks_per_beat // self.config.max_num_pos_per_beat
            elif event.type_ == "Tempo":
                current_tempo = event.value
            elif event.type_ == "Program":
                current_program = event.value
            elif event.type_ in {"Pitch", "PitchDrum"} and e + 2 < len(events):
                pitch_token_name = (
                    "PitchDrum" if event.type_ == "PitchDrum" else "Pitch"
                )
                IOI = current_bar * pos_per_bar + current_pos - prev_bar * pos_per_bar - prev_pos if prev_bar != -1 else 0
                
                if self.config.additional_params["data_type"] == "Performance":
                    new_event = [
                        Event(type_=pitch_token_name, value=event.value, time=event.time),
                        Event(type_="PVelocity", value=events[e + 1].value, time=event.time),
                        Event(type_="PDuration", value=events[e + 2].value, time=event.time),
                        Event(type_="PIOI", value=IOI, time=event.time ),
                        Event(type_="PPosition", value=current_pos, time=event.time),
                        Event(type_="PBar", value=current_bar, time=event.time),
                    ]
                else:
                    new_event = [
                        Event(type_=pitch_token_name, value=event.value, time=event.time),
                        Event(type_="SDuration", value=events[e + 1].value, time=event.time),
                        Event(type_="SIOI", value=IOI, time=event.time ),
                        Event(type_="SPosition", value=current_pos, time=event.time),
                        Event(type_="SBar", value=current_bar, time=event.time),
                    ]
                if self.config.use_programs:
                    new_event.append(Event("Program", current_program))
                if self.config.use_tempos:
                    new_event.append(Event(type_="Tempo", value=current_tempo))
                if self.config.use_time_signatures:
                    new_event.append(
                        Event(
                            type_="TimeSig",
                            value=f"{current_time_sig[0]}/{current_time_sig[1]}",
                        )
                    )
                all_events.append(new_event)

        return all_events

    def _midi_to_tokens(self, midi: Score) -> TokSequence | list[TokSequence]:
        r"""
        Convert a **preprocessed** MIDI object to a sequence of tokens.

        We override the parent method in order to check the number of bars in the MIDI.
        The workflow of this method is as follows: the global events (*Tempo*,
        *TimeSignature*...) and track events (*Pitch*, *Velocity*, *Pedal*...) are
        gathered into a list, then the time events are added. If `one_token_stream` is
        ``True``, all events of all tracks are treated all at once, otherwise the
        events of each track are treated independently.

        :param midi: the MIDI :class:`symusic.Score` object to convert.
        :return: a :class:`miditok.TokSequence` if ``tokenizer.one_token_stream`` is
            ``True``, else a list of :class:`miditok.TokSequence` objects.
        """
        self.ticks_per_quarter = midi.ticks_per_quarter
        
        # Check bar embedding limit, update if needed
        num_bars = len(get_bars_ticks(midi))
        if self.config.additional_params["max_bar_embedding"] < num_bars:
            for i in range(
                self.config.additional_params["max_bar_embedding"], num_bars
            ):
                self.add_to_vocab(f"Bar_{i}", 4)
            self.config.additional_params["max_bar_embedding"] = num_bars

        return super()._midi_to_tokens(midi)
    
    def _create_track_events(
        self, track: Track, ticks_per_beat: np.ndarray = None
    ) -> list[Event]:
        r"""
        Extract the tokens/events from a track (``symusic.Track``).

        Concerned events are: *Pitch*, *Velocity*, *Duration*, *NoteOn*, *NoteOff* and
        optionally *Chord*, *Pedal* and *PitchBend*.
        **If the tokenizer is using pitch intervals, the notes must be sorted by time
        then pitch values. This is done in** ``preprocess_midi``.

        :param track: ``symusic.Track`` to extract events from.
        :param ticks_per_beat: array indicating the number of ticks per beat per
            section. The numbers of ticks per beat depend on the time signatures of
            the MIDI being parsed. The array has a shape ``(N,2)``, for ``N`` changes
            of ticks per beat, and the second dimension representing the end tick of
            each portion and the number of ticks per beat respectively.
            This argument is not required if the tokenizer is not using *Duration*,
            *PitchInterval* or *Chord* tokens. (default: ``None``)
        :return: sequence of corresponding ``Event``s.
        """
        program = track.program if not track.is_drum else -1
        events = []
        # max_time_interval is adjusted depending on the time signature denom / tpb
        max_time_interval = 0
        if self.config.use_pitch_intervals:
            max_time_interval = (
                ticks_per_beat[0, 1] * self.config.pitch_intervals_max_time_dist
            )
        previous_note_onset = -max_time_interval - 1
        previous_pitch_onset = -128  # lowest at a given time
        previous_pitch_chord = -128  # for chord intervals

        # Add sustain pedal
        if self.config.use_sustain_pedals:
            tpb_idx = 0
            for pedal in track.pedals:
                # If not using programs, the default value is 0
                events.append(
                    Event(
                        "Pedal",
                        program if self.config.use_programs else 0,
                        pedal.time,
                        program,
                    )
                )
                # PedalOff or Duration
                if self.config.sustain_pedal_duration:
                    # `while` here as there might not be any note in the next section
                    while pedal.time >= ticks_per_beat[tpb_idx, 0]:
                        tpb_idx += 1
                    dur = self._tpb_ticks_to_tokens[ticks_per_beat[tpb_idx, 1]][
                        pedal.duration
                    ]
                    events.append(
                        Event(
                            "Duration",
                            dur,
                            pedal.time,
                            program,
                            "PedalDuration",
                        )
                    )
                else:
                    events.append(Event("PedalOff", program, pedal.end, program))

        # Add pitch bend
        if self.config.use_pitch_bends:
            for pitch_bend in track.pitch_bends:
                if self.config.use_programs and not self.config.program_changes:
                    events.append(
                        Event(
                            "Program",
                            program,
                            pitch_bend.time,
                            program,
                            "ProgramPitchBend",
                        )
                    )
                events.append(
                    Event("PitchBend", pitch_bend.value, pitch_bend.time, program)
                )

        # Control changes (in the future, and handle pedals redundancy)

        # Add chords
        if self.config.use_chords and not track.is_drum:
            chords = detect_chords(
                track.notes,
                ticks_per_beat,
                chord_maps=self.config.chord_maps,
                program=program,
                specify_root_note=self.config.chord_tokens_with_root_note,
                beat_res=self._first_beat_res,
                unknown_chords_num_notes_range=self.config.chord_unknown,
            )
            for chord in chords:
                if self.config.use_programs and not self.config.program_changes:
                    events.append(
                        Event("Program", program, chord.time, program, "ProgramChord")
                    )
                events.append(chord)

        # Creates the Note On, Note Off and Velocity events
        tpb_idx = 0
        for note in track.notes:
            # Program
            if self.config.use_programs and not self.config.program_changes:
                events.append(
                    Event(
                        type_="Program",
                        value=program,
                        time=note.start,
                        program=program,
                        desc=note.end,
                    )
                )

            # Pitch interval
            add_absolute_pitch_token = True
            if self.config.use_pitch_intervals and not track.is_drum:
                # Adjust max_time_interval if needed
                if note.time >= ticks_per_beat[tpb_idx, 0]:
                    tpb_idx += 1
                    max_time_interval = (
                        ticks_per_beat[tpb_idx, 1]
                        * self.config.pitch_intervals_max_time_dist
                    )
                if note.start != previous_note_onset:
                    if (
                        note.start - previous_note_onset <= max_time_interval
                        and abs(note.pitch - previous_pitch_onset)
                        <= self.config.max_pitch_interval
                    ):
                        events.append(
                            Event(
                                type_="PitchIntervalTime",
                                value=note.pitch - previous_pitch_onset,
                                time=note.start,
                                program=program,
                                desc=note.end,
                            )
                        )
                        add_absolute_pitch_token = False
                    previous_pitch_onset = previous_pitch_chord = note.pitch
                else:  # same onset time
                    if (
                        abs(note.pitch - previous_pitch_chord)
                        <= self.config.max_pitch_interval
                    ):
                        events.append(
                            Event(
                                type_="PitchIntervalChord",
                                value=note.pitch - previous_pitch_chord,
                                time=note.start,
                                program=program,
                                desc=note.end,
                            )
                        )
                        add_absolute_pitch_token = False
                    else:
                        # We update previous_pitch_onset as there might be a chord
                        # interval starting from the current note to the next one.
                        previous_pitch_onset = note.pitch
                    previous_pitch_chord = note.pitch
                previous_note_onset = note.start

            # Pitch / NoteOn
            if add_absolute_pitch_token:
                if self.config.use_pitchdrum_tokens and track.is_drum:
                    note_token_name = "DrumOn" if self._note_on_off else "PitchDrum"
                else:
                    note_token_name = "NoteOn" if self._note_on_off else "Pitch"
                events.append(
                    Event(
                        type_=note_token_name,
                        value=note.pitch,
                        time=note.start,
                        program=program,
                        desc=note.end,
                    )
                )
            
            ###### ADD BY JINGJING ##########
            if self.config.additional_params["data_type"] == "Performance":
                # Velocity
                events.append(
                    Event(
                        type_="PVelocity",
                        value=note.velocity,
                        time=note.start,
                        program=program,
                        desc=f"{note.velocity}",
                    )
                )

            # Duration / NoteOff
            if self._note_on_off:
                if self.config.use_programs and not self.config.program_changes:
                    events.append(
                        Event(
                            type_="Program",
                            value=program,
                            time=note.end,
                            program=program,
                            desc="ProgramNoteOff",
                        )
                    )
                events.append(
                    Event(
                        type_="DrumOff"
                        if self.config.use_pitchdrum_tokens and track.is_drum
                        else "NoteOff",
                        value=note.pitch,
                        time=note.end,
                        program=program,
                        desc=note.end,
                    )
                )
            else:
                # `while` as there might not be any note in the next section
                while note.time >= ticks_per_beat[tpb_idx, 0]:
                    tpb_idx += 1
                dur = self._tpb_ticks_to_tokens[ticks_per_beat[tpb_idx, 1]][
                    note.duration
                ]
                if self.config.additional_params["data_type"] == "Performance":
                    events.append(
                        Event(
                            type_="PDuration",
                            value=dur,
                            time=note.start,
                            program=program,
                            desc=f"{note.duration} ticks",
                        )
                    )
                else:
                    events.append(
                        Event(
                            type_="SDuration",
                            value=dur,
                            time=note.start,
                            program=program,
                            desc=f"{note.duration} ticks",
                        )
                    )

        return events    
    
    def alignment_to_token(self, alignment_file: str, midi: Score) -> TokSequence | list[TokSequence]:      
        headers = ['alignID', 'alignOntime', 'alignOfftime', 'alignSitch', 'alignPitch', 'alignOnvel', 
                        'refID', 'refOntime', 'refOfftime', 'refSitch', 'refPitch', 'refOnvel']
        align_file = pd.read_csv(alignment_file, sep='\s+', names=headers, skiprows=[0])
        
        align_file['label'] = None
        align_file.loc[align_file['refID']== "*", 'label'] = "insertion"
        align_file.loc[align_file['alignID']== "*", 'label'] = "deletion"
        align_file.loc[(align_file['refID']!= "*")&(align_file['alignID']!= "*"), 'label'] = 'match'
        
        align_file = align_file[align_file['label'] == "match"].astype({'refID':'int32'})
        outliers = self._detect_outliers(align_file)
        align_file = align_file[~align_file['refID'].isin(outliers)]
        
        all_events = []
        # Global events (Tempo, TimeSignature)
        global_events = [Event(type_="TimeSig", value="4/4", time=0),
                         Event(type_="Tempo", value=TEMPO, time=0)]
        
        all_events += global_events       
        ticks_per_beat = TICKS_PER_BEAT

        # Adds track tokens
        all_events += self._create_align_events(align_file, midi)
        self._sort_events(all_events)
        # Add time events
        all_events = self._add_align_time_events(all_events, midi.ticks_per_quarter)
        tok_sequence = TokSequence(events=all_events)
        self.complete_sequence(tok_sequence)
  
        return tok_sequence
    
    def _create_align_events(self, alignment:pd.DataFrame, midi: Score) -> list[Event]:
        events = []
        previous_Ponset = 0
        previous_Sonset = 0
        ticks_per_beat = get_midi_ticks_per_beat(midi)
        alignment.loc[:, 'alignOnvel'] = self.np_get_closest(self.velocities, alignment['alignOnvel'].to_numpy())
        
        for idx, row in alignment.iterrows():
            row['refOfftime'] = row['refOfftime'] + 0.5 if row['refOfftime'] == row['refOntime'] else row['refOfftime']
            
            Ponset = self._seconds_to_ticks(row['alignOntime'], ticks_per_beat[0, 1])
            Poffset = self._seconds_to_ticks(row['alignOfftime'], ticks_per_beat[0, 1])
            Sonset = self._seconds_to_ticks(row['refOntime'], ticks_per_beat[0, 1])
            Soffset = self._seconds_to_ticks(row['refOfftime'], ticks_per_beat[0, 1])
            
            
            # Pitch
            events.append(
                Event(
                    type_="Pitch",
                    value=row['alignPitch'],
                    time=Ponset,
                    desc=Poffset
                )
            )

            #ScoreOnset
            events.append(
                Event(
                    type_="SOnset",
                    value=Sonset,
                    time=Ponset,
                    desc=Poffset
                )
            )

            
            # PVelocity
            events.append(
                Event(
                    type_="PVelocity",
                    value=row['alignOnvel'],
                    time=Ponset,
                    desc=f"{row['alignOnvel']}"
                )
            )
            
            # PDuration
            Pdur = Poffset - Ponset
            Pdur = self._tpb_ticks_to_tokens[ticks_per_beat[0, 1]][
                    Pdur
                ]
            events.append(
                    Event(
                        type_="PDuration",
                        value=Pdur,
                        time=Ponset,
                        desc=f"{Pdur} ticks",
                    )
                )

            # SDuration
            Sdur = Soffset - Sonset
            Sdur = self._tpb_ticks_to_tokens[ticks_per_beat[0, 1]][
                    Sdur
                ] 
            events.append(
                    Event(
                        type_="SDuration",
                        value=Sdur,
                        time=Ponset,
                        desc=f"{Sdur} ticks",
                    )
                )
            
        return events
    
    def _add_align_time_events(self, events: list[Event], time_division: int
    ) -> list[list[Event]]:
        r"""
        Create the time events from a list of performance and score events.

        Internal method intended to be implemented by child classes.
        The returned sequence is the final token sequence ready to be converted to ids
        to be fed to a model.

        :param events: sequence of global and track events to create tokens time from.
        :param time_division: time division in ticks per quarter of the MIDI being
            tokenized.
        :return: the same events, with time events inserted.
        """
        # Add time events
        all_events = []
        # Performance
        Pcurrent_bar = 0
        Pcurrent_bar_from_ts_time = 0
        Pcurrent_tick_from_ts_time = 0
        Pcurrent_pos = 0
        Pprevious_tick = 0
        Pprev_bar = -1
        Pprev_pos = 0
               
        
        # Score
        Scurrent_bar = 0
        Scurrent_bar_from_ts_time = 0
        Scurrent_tick_from_ts_time = 0
        Scurrent_pos = 0
        Sprevious_tick = 0
        Sprev_bar = -1
        Sprev_pos = 0
        
        current_time_sig = TIME_SIGNATURE
        ticks_per_bar = compute_ticks_per_bar(
            TimeSignature(0, *current_time_sig), time_division
        )
        ticks_per_beat = compute_ticks_per_beat(current_time_sig[1], time_division)
        ticks_per_pos = ticks_per_beat // self.config.max_num_pos_per_beat
        pos_per_bar = max(self.config.beat_res.values())

        for e, event in enumerate(events):
            # Set current bar and position
            # This is done first, as we need to compute these values with the current
            # ticks_per_bar, which might change if the current event is a TimeSig
            if (event.time != Pprevious_tick) & (event.type_ != "SOnset"):
                Pelapsed_tick = event.time - Pcurrent_tick_from_ts_time
                Pcurrent_bar = Pcurrent_bar_from_ts_time + Pelapsed_tick // ticks_per_bar
                Ptick_at_current_bar = (
                    Pcurrent_tick_from_ts_time
                    + (Pcurrent_bar - Pcurrent_bar_from_ts_time) * ticks_per_bar
                )
                Pcurrent_pos = (event.time - Ptick_at_current_bar) // ticks_per_pos
                Pprevious_tick = event.time
            
            if (event.value != Sprevious_tick) & (event.type_ == "SOnset"):
                Selapsed_tick = event.value - Scurrent_tick_from_ts_time
                Scurrent_bar = Scurrent_bar_from_ts_time + Selapsed_tick // ticks_per_bar
                Stick_at_current_bar = (
                    Scurrent_tick_from_ts_time
                    + (Scurrent_bar - Scurrent_bar_from_ts_time) * ticks_per_bar
                )
                Scurrent_pos = (event.value - Stick_at_current_bar) // ticks_per_pos
                Sprevious_tick = event.value

            if event.type_ == "TimeSig":
                current_time_sig = list(map(int, event.value.split("/")))
                Pcurrent_bar_from_ts_time = Pcurrent_bar
                Pcurrent_tick_from_ts_time = Pprevious_tick
                Scurrent_bar_from_ts_time = Scurrent_bar
                Scurrent_tick_from_ts_time = Sprevious_tick
                ticks_per_bar = compute_ticks_per_bar(
                    TimeSignature(event.time, *current_time_sig), time_division
                )
                ticks_per_beat = compute_ticks_per_beat(
                    current_time_sig[1], time_division
                )
                ticks_per_pos = ticks_per_beat // self.config.max_num_pos_per_beat
            elif event.type_ in {"Pitch"} and e + 5 < len(events):
                pitch_token_name = "Pitch"
                # "Pitch", "PVelocity", "PDuration", "PIOI", "PPosition", "PBar", 
                # "SDuration", "SIOI", "SPosition", "SBar", "SPDurationDev"
                PIOI = Pcurrent_bar * pos_per_bar + Pcurrent_pos - Pprev_bar * pos_per_bar - Pprev_pos if Pprev_bar != -1 else 0
                SIOI = Scurrent_bar * pos_per_bar + Scurrent_pos - Sprev_bar * pos_per_bar - Sprev_pos if Sprev_bar != -1 else 0
                Pdur = int(events[e + 3].value.split(".")[0]) * int(events[e + 3].value.split(".")[2]) + int(events[e + 3].value.split(".")[1])
                Sdur = int(events[e + 4].value.split(".")[0]) * int(events[e + 4].value.split(".")[2]) + int(events[e + 4].value.split(".")[1])
                SPDurDev = Pdur - Sdur
                
                if Pprev_bar == -1:
                    print(PIOI)
                    assert PIOI == 0
                
                
                if Sprev_bar == -1:
                    assert SIOI == 0
                
                new_event = [
                    Event(type_=pitch_token_name, value=event.value, time=event.time),
                    Event(type_="PVelocity", value=events[e + 2].value, time=event.time),
                    Event(type_="PDuration", value=events[e + 3].value, time=event.time),
                    Event(type_="PIOI", value=PIOI, time=event.time),
                    Event(type_="PPosition", value=Pcurrent_pos, time=event.time),
                    Event(type_="PBar", value=Pcurrent_bar, time=event.time),
                    Event(type_="SDuration", value=events[e + 4].value, time=event.time),
                    Event(type_="SIOI", value=SIOI, time=event.time),
                    Event(type_="SPosition", value=Scurrent_pos, time=event.time),
                    Event(type_="SBar", value=Scurrent_bar, time=event.time),
                    Event(type_="SPDurationDev", value=SPDurDev, time=event.time),
                ]
                
                all_events.append(new_event)
                Pprev_bar = Pcurrent_bar
                Pprev_pos = Pcurrent_pos
                Sprev_bar = Scurrent_bar
                Sprev_pos = Scurrent_pos

        return all_events

    def _align_tokens_to_midi(
        self,
        tokens: TokSequence | list[TokSequence],
        programs: list[tuple[int, bool]] | None = None,
    ) -> list[Score]:
        r"""
        Convert tokens (:class:`miditok.TokSequence`) into a MIDI.

        This is an internal method called by ``self.tokens_to_midi``, intended to be
        implemented by classes inheriting :class:`miditok.MidiTokenizer`.

        :param tokens: tokens to convert. Can be either a list of
            :class:`miditok.TokSequence` or a list of :class:`miditok.TokSequence`s.
        :param programs: programs of the tracks. If none is given, will default to
            piano, program 0. (default: ``None``)
        :return: the midi object (:class:`symusic.Score`).
        """
        # Unsqueeze tokens in case of one_token_stream
        tokens = [tokens]
        Pmidi = Score(self.time_division)
        Smidi = Score(self.time_division)

        # RESULTS
        Ptracks: dict[int, Track] = {}
        Stracks: dict[int, Track] = {}
        
        tempo_changes, time_signature_changes = [Tempo(-1, self.default_tempo)], []
        tempo_changes[0].tempo = -1

        def check_inst(prog: int, tracks: dict[int, Track]) -> None:
            if prog not in tracks:
                tracks[prog] = Track(
                    program=0 if prog == -1 else prog,
                    is_drum=prog == -1,
                    name="Drums" if prog == -1 else MIDI_INSTRUMENTS[prog]["name"],
                )

        bar_at_last_ts_change = 0
        tick_at_last_ts_change = 0
        current_program = 0 #Piano

        for si, seq in enumerate(tokens):
            # First look for the first time signature if needed
            time_signature_changes.append(TimeSignature(0, *TIME_SIGNATURE))
            current_time_sig = time_signature_changes[0]
            ticks_per_bar = compute_ticks_per_bar(
                current_time_sig, self.ticks_per_quarter
            )
            ticks_per_beat = self._tpb_per_ts[current_time_sig.denominator]
            ticks_per_pos = ticks_per_beat // self.config.max_num_pos_per_beat
                
            # Decode tokens
            for time_step in seq:
                num_tok_to_check = 11
                if any(
                    tok.split("_")[1] == "None" for tok in time_step[:num_tok_to_check]
                ):
                    # Padding or mask: error of prediction or end of sequence anyway
                    continue

                # Note attributes
                pitch = int(time_step[0].split("_")[1])
                Pvel = int(time_step[1].split("_")[1])

                # Time values
                Pevent_pos = int(time_step[4].split("_")[1])
                Pevent_bar = int(time_step[5].split("_")[1])
                Pcurrent_tick = (
                    tick_at_last_ts_change
                    + (Pevent_bar - bar_at_last_ts_change) * ticks_per_bar
                    + Pevent_pos * ticks_per_pos
                )

                Sevent_pos = int(time_step[8].split("_")[1])
                Sevent_bar = int(time_step[9].split("_")[1])
                Scurrent_tick = (
                    tick_at_last_ts_change
                    + (Sevent_bar - bar_at_last_ts_change) * ticks_per_bar
                    + Sevent_pos * ticks_per_pos
                )


                # Note duration
                Pduration = self._tpb_tokens_to_ticks[ticks_per_beat][
                    time_step[2].split("_")[1]
                ]
                
                Sduration = self._tpb_tokens_to_ticks[ticks_per_beat][
                    time_step[6].split("_")[1]
                ]

                # Append the created note
                new_Pnote = Note(Pcurrent_tick, Pduration, pitch, Pvel)
                # Set the velocity for scores to be constant 60
                new_Snote = Note(Scurrent_tick, Sduration, pitch, 60) 
              
                check_inst(current_program, Ptracks)
                check_inst(current_program, Stracks)
                Ptracks[current_program].notes.append(new_Pnote)
                Stracks[current_program].notes.append(new_Snote)

        # create MidiFile
        Pmidi.tracks = list(Ptracks.values())
        Pmidi.tempos = tempo_changes
        Pmidi.time_signatures = time_signature_changes
        
        Smidi.tracks = list(Stracks.values())
        Smidi.tempos = tempo_changes
        Smidi.time_signatures = time_signature_changes

        return [Pmidi, Smidi]
            
    def _tokens_to_midi(
        self,
        tokens: TokSequence | list[TokSequence],
        programs: list[tuple[int, bool]] | None = None,
    ) -> Score:
        r"""
        Convert tokens (:class:`miditok.TokSequence`) into a MIDI.

        This is an internal method called by ``self.tokens_to_midi``, intended to be
        implemented by classes inheriting :class:`miditok.MidiTokenizer`.

        :param tokens: tokens to convert. Can be either a list of
            :class:`miditok.TokSequence` or a list of :class:`miditok.TokSequence`s.
        :param programs: programs of the tracks. If none is given, will default to
            piano, program 0. (default: ``None``)
        :return: the midi object (:class:`symusic.Score`).
        """
        # Unsqueeze tokens in case of one_token_stream
        if self.one_token_stream:  # ie single token seq
            tokens = [tokens]
        for i in range(len(tokens)):
            tokens[i] = tokens[i].tokens
        midi = Score(self.time_division)

        # RESULTS
        tracks: dict[int, Track] = {}
        tempo_changes, time_signature_changes = [Tempo(-1, self.default_tempo)], []
        tempo_changes[0].tempo = -1

        def check_inst(prog: int) -> None:
            if prog not in tracks:
                tracks[prog] = Track(
                    program=0 if prog == -1 else prog,
                    is_drum=prog == -1,
                    name="Drums" if prog == -1 else MIDI_INSTRUMENTS[prog]["name"],
                )

        def is_track_empty(track: Track) -> bool:
            return (
                len(track.notes) == len(track.controls) == len(track.pitch_bends) == 0
            )

        bar_at_last_ts_change = 0
        tick_at_last_ts_change = 0
        current_program = 0
        current_track = None
        for si, seq in enumerate(tokens):
            # First look for the first time signature if needed
            if si == 0 and self.config.use_time_signatures:
                num, den = self._parse_token_time_signature(
                    seq[0][self.vocab_types_idx["TimeSig"]].split("_")[1]
                )
                time_signature_changes.append(TimeSignature(0, num, den))
            else:
                time_signature_changes.append(TimeSignature(0, *TIME_SIGNATURE))
            current_time_sig = time_signature_changes[0]
            ticks_per_bar = compute_ticks_per_bar(
                current_time_sig, midi.ticks_per_quarter
            )
            ticks_per_beat = self._tpb_per_ts[current_time_sig.denominator]
            ticks_per_pos = ticks_per_beat // self.config.max_num_pos_per_beat
            # Set track / sequence program if needed
            if not self.one_token_stream:
                is_drum = False
                if programs is not None:
                    current_program, is_drum = programs[si]
                current_track = Track(
                    program=current_program,
                    is_drum=is_drum,
                    name="Drums"
                    if current_program == -1
                    else MIDI_INSTRUMENTS[current_program]["name"],
                )

            # Decode tokens
            for time_step in seq:
                num_tok_to_check = 6 if self.config.use_programs else 5
                if any(
                    tok.split("_")[1] == "None" for tok in time_step[:num_tok_to_check]
                ):
                    # Padding or mask: error of prediction or end of sequence anyway
                    continue

                # Note attributes
                pitch = int(time_step[0].split("_")[1])
                if self.config.additional_params["data_type"] == "Performance":
                    vel = int(time_step[1].split("_")[1])
                else:
                    vel = 60
                # if self.config.use_programs:
                #     current_program = int(time_step[5].split("_")[1])

                if self.config.additional_params["data_type"] == "Performance":
                    # Time values
                    event_pos = int(time_step[4].split("_")[1])
                    event_bar = int(time_step[5].split("_")[1])
                else:
                    event_pos = int(time_step[3].split("_")[1])
                    event_bar = int(time_step[4].split("_")[1])
                
                current_tick = (
                    tick_at_last_ts_change
                    + (event_bar - bar_at_last_ts_change) * ticks_per_bar
                    + event_pos * ticks_per_pos
                )

                # Time Signature, adds a TimeSignatureChange if necessary
                if (
                    self.config.use_time_signatures
                    and time_step[self.vocab_types_idx["TimeSig"]].split("_")[1]
                    != "None"
                ):
                    num, den = self._parse_token_time_signature(
                        time_step[self.vocab_types_idx["TimeSig"]].split("_")[1]
                    )
                    if (
                        num != current_time_sig.numerator
                        or den != current_time_sig.denominator
                    ):
                        # tick from bar of ts change
                        tick_at_last_ts_change += (
                            event_bar - bar_at_last_ts_change
                        ) * ticks_per_bar
                        current_time_sig = TimeSignature(
                            tick_at_last_ts_change, num, den
                        )
                        if si == 0:
                            time_signature_changes.append(current_time_sig)
                        bar_at_last_ts_change = event_bar
                        ticks_per_bar = compute_ticks_per_bar(
                            current_time_sig, midi.ticks_per_quarter
                        )
                        ticks_per_beat = self._tpb_per_ts[current_time_sig.denominator]
                        ticks_per_pos = (
                            ticks_per_beat // self.config.max_num_pos_per_beat
                        )

                # Note duration
                if self.config.additional_params["data_type"] == "Performance":
                    duration = self._tpb_tokens_to_ticks[ticks_per_beat][
                        time_step[2].split("_")[1]
                    ]
                else:
                    duration = self._tpb_tokens_to_ticks[ticks_per_beat][
                        time_step[1].split("_")[1]
                    ]

                # Append the created note
                new_note = Note(current_tick, duration, pitch, vel)
                if self.one_token_stream:
                    check_inst(current_program)
                    tracks[current_program].notes.append(new_note)
                else:
                    current_track.notes.append(new_note)

                # Tempo, adds a TempoChange if necessary
                if (
                    si == 0
                    and self.config.use_tempos
                    and time_step[self.vocab_types_idx["Tempo"]].split("_")[1] != "None"
                ):
                    tempo = float(
                        time_step[self.vocab_types_idx["Tempo"]].split("_")[1]
                    )
                    if tempo != round(tempo_changes[-1].tempo, 2):
                        tempo_changes.append(Tempo(current_tick, tempo))

            # Add current_inst to midi and handle notes still active
            if not self.one_token_stream and not is_track_empty(current_track):
                midi.tracks.append(current_track)

        # Delete mocked
        # And handle first tempo (tick 0) here instead of super
        del tempo_changes[0]
        if len(tempo_changes) == 0 or (
            tempo_changes[0].time != 0
            and round(tempo_changes[0].tempo, 2) != self.default_tempo
        ):
            tempo_changes.insert(0, Tempo(0, self.default_tempo))
        elif round(tempo_changes[0].tempo, 2) == self.default_tempo:
            tempo_changes[0].time = 0

        # create MidiFile
        if self.one_token_stream:
            midi.tracks = list(tracks.values())
        midi.tempos = tempo_changes
        midi.time_signatures = time_signature_changes

        return midi

    def _create_base_vocabulary(self) -> list[list[str]]:
        r"""
        Create the vocabulary, as a list of string tokens.

        Each token is given as the form ``"Type_Value"``, with its type and value
        separated with an underscore. Example: ``Pitch_58``.
        The :class:`miditok.MIDITokenizer` main class will then create the "real"
        vocabulary as a dictionary. Special tokens have to be given when creating the
        tokenizer, and will be added to the vocabulary by
        :class:`miditok.MIDITokenizer`.

        :return: the vocabulary as a list of string.
        """
        if self.config.additional_params["data_type"] == "Performance":
            N = 6
        elif self.config.additional_params["data_type"] == "Score":
            N = 5
        else:
            N = 11 
            
        max_num_beats = max(ts[0] for ts in self.time_signatures)
        num_positions = self.config.max_num_pos_per_beat * max_num_beats
        
        vocab = [[] for _ in range(N)]
        
        # PITCH
        vocab[0] += [f"Pitch_{i}" for i in range(*self.config.pitch_range)]
        if self.config.additional_params["data_type"] in {"Performance", "Alignment"}:
            # PVELOCITY
            vocab[1] += [f"PVelocity_{i}" for i in self.velocities]

            # PDURATION
            vocab[2] += [
                f'PDuration_{".".join(map(str, duration))}' for duration in self.durations
            ]

            # PPOSITION & PIOI
            # self.time_division is equal to the maximum possible ticks/beat value.
            vocab[3] += [f"PIOI_{i}" for i in range(-num_positions, num_positions)]
            vocab[4] += [f"PPosition_{i}" for i in range(num_positions)]

            # PBAR (positional encoding)
            vocab[5] += [
                f"PBar_{i}"
                for i in range(self.config.additional_params["max_bar_embedding"])
            ]
            
        if self.config.additional_params["data_type"] in {"Score", "Alignment"}:
            n = 5 if self.config.additional_params["data_type"] == "Alignment" else 0
            # SDURATION
            vocab[n + 1] += [
                f'SDuration_{".".join(map(str, duration))}' for duration in self.durations
            ]

            # SPOSITION & SIOI
            # self.time_division is equal to the maximum possible ticks/beat value.
            vocab[n + 2] += [f"SIOI_{i}" for i in range(-num_positions, num_positions)]
            
            vocab[n + 3] += [f"SPosition_{i}" for i in range(num_positions)]

            # SBAR (positional encoding)
            vocab[n + 4] += [
                f"SBar_{i}"
                for i in range(self.config.additional_params["max_bar_embedding"])
            ]
            
        if self.config.additional_params["data_type"] == "Alignment":
            # SDURATION
            vocab[10] += [
                f'SPDurationDev_{i}' for i in range(-2*num_positions, 2*num_positions)
            ]
        return vocab

    def _create_token_types_graph(self) -> dict[str, set[str]]:
        r"""
        Return a graph/dictionary of the possible token types successions.

        Not relevant for Octuple as it is not subject to token type errors.

        :return: the token types transitions dictionary.
        """
        return {}

    def _tokens_errors(self, tokens: list[list[str]]) -> int:
        r"""
        Return the number of errors in a sequence of tokens.

        The method checks if a sequence of tokens is made of good token types
        successions and values. The number of errors should not be higher than the
        number of tokens.

        The token types are always the same in Octuple so this method only checks
        if their values are correct:
            - a bar token value cannot be < to the current bar (it would go back in
                time)
            - same for positions
            - a pitch token should not be present if the same pitch is already played
                at the current position.

        :param tokens: sequence of tokens string to check.
        :return: the number of errors predicted (no more than one per token).
        """
        err = 0
        current_bar = current_pos = -1
        current_pitches = {p: [] for p in self.config.programs}
        current_program = 0

        for token in tokens:
            if any(tok.split("_")[1] == "None" for tok in token):
                err += 1
                continue
            has_error = False
            bar_value = int(token[5].split("_")[1])
            pos_value = int(token[4].split("_")[1])
            pitch_value = int(token[0].split("_")[1])
            if self.config.use_programs:
                current_program = int(token[5].split("_")[1])

            # Bar
            if bar_value < current_bar:
                has_error = True
            elif bar_value > current_bar:
                current_bar = bar_value
                current_pos = -1
                current_pitches = {p: [] for p in self.config.programs}

            # Position
            if pos_value < current_pos:
                has_error = True
            elif pos_value > current_pos:
                current_pos = pos_value
                current_pitches = {p: [] for p in self.config.programs}

            # Pitch
            if self.config.remove_duplicated_notes:
                if pitch_value in current_pitches[current_program]:
                    has_error = True
                else:
                    current_pitches[current_program].append(pitch_value)

            if has_error:
                err += 1

        return err
    
    def _seconds_to_ticks(self, seconds, ticks_per_beat=TICKS_PER_BEAT, tempo=TEMPO):
        """
        Converts time in seconds to MIDI ticks.
        
        Args:
        seconds (float): Time in seconds.
        tempo (int): Tempo in beats per minute (BPM).
        ticks_per_beat (int): Resolution of the MIDI file, in ticks per beat.
        
        Returns:
        int: Number of ticks corresponding to the number of seconds.
        """
        # Calculate the duration of a single beat in seconds
        seconds_per_beat = 60.0 / tempo
        
        # Calculate the number of beats in the given number of seconds
        beats = seconds / seconds_per_beat
        
        # Convert beats to ticks
        ticks = int(beats * ticks_per_beat)
        
        return ticks
    
    def _detect_outliers(self, align_file):
        """Detect disordered score notes (time not in ascending order)

        Args:
            align_file (pd.Dataframe): _description_

        Returns:
            List: list of the index for outliers
        """
        outliers = []
        sequence = align_file['refID'].tolist()
        onset = align_file['refOntime'].tolist()
        for i in range(1, len(sequence)-2):
            if ((np.abs(sequence[i] - sequence[i-1]) > 10) or (np.abs(sequence[i] - sequence[i+1]) > 10)) and \
                ((np.abs(onset[i] - onset[i-1]) > 1) or (np.abs(onset[i] - onset[i+1]) > 1)):
                if np.abs(sequence[i+2] + sequence[i+1] - 2*sequence[i]) > 20:
                    outliers.append(sequence[i])
        return outliers
    
    @staticmethod
    def np_get_closest(array: np.ndarray, values: np.ndarray) -> np.ndarray:
        """
        Find the closest values to those of another reference array.

        Taken from: https://stackoverflow.com/a/46184652.

        :param array: reference values array.
        :param values: array to filter.
        :return: the closest values for each position.
        """
        # get insert positions
        idxs = np.searchsorted(array, values, side="left")

        # find indexes where previous index is closer
        prev_idx_is_less = (idxs == len(array)) | (
            np.fabs(values - array[np.maximum(idxs - 1, 0)])
            < np.fabs(values - array[np.minimum(idxs, len(array) - 1)])
        )
        idxs[prev_idx_is_less] -= 1

        return array[idxs]

if __name__ == "__main__":
    
    """
    Expression tokenizer.

    * 0: Pitch;
    * 1: Performance Velocity; -> PVelocity
    * 2: Performance Duration; -> PDuration
    * 3: Performance Inter Onset Interval (Onset time difference between the current note and the previous note); -> PIOI
    * 4: Perfromance Position; -> PPosition
    * 5: Perfromance Bar; -> PBar
    * 6: Score Duration; -> SDuration
    * 7: Score Inter Onset Interval; -> SIOI
    * 8: Score Position; -> SPosition
    * 9: Score Bar; -> SBar
    * 10: Duration Deviation; -> SPDurationDev

    **Notes:**
    * Tokens are first sorted by time, then track, then pitch values.
    
    """
    
    DATA_FOLDER = "/home/smg/v-jtbetsy/DATA/ATEPP-s2a"
    align_path = "Ludwig_van_Beethoven/Piano_Sonata_No._7_in_D_Major,_Op._10_No._3/III._Menuetto._Allegro/05813_infer_corresp.txt"
    performance_midi_path = "Ludwig_van_Beethoven/Piano_Sonata_No._7_in_D_Major,_Op._10_No._3/III._Menuetto._Allegro/05813.mid"
    score_midi_path = "Ludwig_van_Beethoven/Piano_Sonata_No._7_in_D_Major,_Op._10_No._3/III._Menuetto._Allegro/musicxml_cleaned.musicxml.midi"
    
    TOKENIZER_PARAMS = {
        "pitch_range": (21, 109),
        "beat_res": {(0, 12):TICKS_PER_BEAT},
        "num_velocities": 64,
        "special_tokens": ["PAD", "BOS", "EOS", "MASK"],
        "use_chords": False,
        "use_rests": False,
        "use_tempos": False,
        "use_time_signatures": False,
        "use_programs": False,
        "num_tempos": 32,  # number of tempo bins
        "tempo_range": (40, 250),  # (min, max)
        "data_type": "Alignment"
    }
    
    
    config = TokenizerConfig(**TOKENIZER_PARAMS)
    
    tokenizer = ExpressionTok(
        config
    )
    
    performance_midi = Score(os.path.join(DATA_FOLDER, performance_midi_path))
    tokens = tokenizer.alignment_to_token(os.path.join(DATA_FOLDER, align_path), performance_midi)
    # print(tokenizer.vocab[3])
    print(tokens[0])
    