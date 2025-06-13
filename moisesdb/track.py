# dftpe_ext/moisesdb/track.py
import logging
import os
import librosa
import numpy as np
import soundfile as sf # Keep for save_audio if used

from .activity import compute_activity_signal
from .defaults import all_stems, default_data_path
from .utils import load_audio, load_json, save_audio

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s (PID:%(process)d)')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)

class MoisesDBTrack:
    _debug_print_counter = 0

    def __init__(
        self, provider, track_id, data_path=default_data_path, sample_rate=44100
    ):
        self.data_path_root = data_path
        self.provider = provider
        self.id = track_id
        self.path = os.path.join(self.data_path_root, self.provider, self.id)
        json_path = os.path.join(self.path, "data.json")
        try:
            self.json_data = load_json(json_path)
        except Exception as e:
            logger.error(f"Track {self.id}: Failed to load/parse JSON at {json_path}: {e}")
            self.json_data = {}
        self.sr = sample_rate
        self.artist = self.json_data.get("artist", "")
        self.name = self.json_data.get("song", "untitled")
        self.genre = self.json_data.get("genre", "undefined")
        self.sources = self._parse_sources(self.json_data.get("stems", []))
        self.bleedings = self._parse_bleeding(self.json_data.get("stems", []))

    def _parse_bleeding(self, stems_metadata_list):
        bleeds = {}
        if not isinstance(stems_metadata_list, list): return bleeds
        for s_meta in stems_metadata_list:
            if not isinstance(s_meta, dict): continue
            stem_name = s_meta.get("stemName")
            if not stem_name: continue
            bleeds.setdefault(stem_name, {})
            tracks_meta = s_meta.get("tracks", [])
            if not isinstance(tracks_meta, list): continue
            for track_component_meta in tracks_meta:
                if not isinstance(track_component_meta, dict): continue
                track_type = track_component_meta.get("trackType")
                if not track_type: continue
                bleeds[stem_name].setdefault(track_type, []).append(track_component_meta.get("has_bleed", False))
        return bleeds

    def _parse_sources(self, stems_metadata_list):
        parsed_stems = {}
        if not isinstance(stems_metadata_list, list): return parsed_stems
        for stem_meta_entry in stems_metadata_list:
            if not isinstance(stem_meta_entry, dict): continue
            stem_name_from_json = stem_meta_entry.get("stemName")
            if not stem_name_from_json: continue
            parsed_stems.setdefault(stem_name_from_json, {})
            current_stem_tracks_metadata = stem_meta_entry.get("tracks", [])
            if not isinstance(current_stem_tracks_metadata, list): continue
            for track_component_meta in current_stem_tracks_metadata:
                if not isinstance(track_component_meta, dict): continue
                component_id = track_component_meta.get("id")
                component_extension = track_component_meta.get("extension")
                track_type = track_component_meta.get("trackType")
                if not all([component_id, component_extension, track_type]): continue
                file_path_abs = os.path.join(self.path, stem_name_from_json, f"{component_id}.{component_extension}")
                parsed_stems[stem_name_from_json].setdefault(track_type, []).append(file_path_abs)
        return parsed_stems

    def get_track_metadata_for_dataset(self):
        duration_s_metadata = self.json_data.get("duration")
        native_sr = self.json_data.get("sample_rate", 44100) # Default to 44100 if not in JSON

        if duration_s_metadata is None: # Try to get duration if not in JSON
            # This is done ONLY when get_track_metadata_for_dataset is called (typically once per track by rank 0)
            first_available_path = None
            if self.sources: # Check if sources were parsed
                for stem_name, track_types in self.sources.items():
                    if first_available_path: break
                    for track_type, paths in track_types.items():
                        if paths and paths[0] and os.path.exists(paths[0]): # Check actual existence
                            first_available_path = paths[0]
                            break
            if first_available_path:
                try:
                    duration_s_metadata = librosa.get_duration(path=first_available_path)
                    # Optionally, cache this back to self.json_data if you want it stored for the object's lifetime
                    # self.json_data["duration"] = duration_s_metadata 
                except Exception as e_get_dur:
                    logger.warning(f"Track {self.id}: Fallback librosa.get_duration FAILED for {first_available_path}: {e_get_dur}. Duration will be None for this track.")
                    duration_s_metadata = None # Ensure it's None
            else:
                logger.warning(f"Track {self.id}: No valid audio paths found in self.sources to determine duration. Duration will be None.")
                duration_s_metadata = None
        
        metadata = {
            "id": self.id,
            "sources_file_paths": self.sources, 
            "sample_rate_native": int(native_sr),
            "duration_seconds_metadata": float(duration_s_metadata) if duration_s_metadata is not None else None
        }
        return metadata

    def _load_and_mix_components(self, component_paths_list, expected_sr):
        if not component_paths_list: return None
        loaded_audios_at_expected_sr = []
        min_len_at_expected_sr = float('inf')
        for audio_path in component_paths_list:
            try:
                audio_arr_native_sr, original_sr = load_audio(audio_path, sr=None) 
                if audio_arr_native_sr is None or audio_arr_native_sr.shape[-1] == 0: continue
                if audio_arr_native_sr.ndim == 1: audio_arr_native_sr = audio_arr_native_sr[np.newaxis, :]
                if audio_arr_native_sr.shape[0] == 0: continue
                if audio_arr_native_sr.shape[0] == 1: audio_arr_native_sr = np.repeat(audio_arr_native_sr, 2, axis=0)
                elif audio_arr_native_sr.shape[0] > 2: audio_arr_native_sr = audio_arr_native_sr[:2, :]
                current_audio_processed = audio_arr_native_sr
                if original_sr != expected_sr:
                    resampled_audio = librosa.resample(audio_arr_native_sr, orig_sr=original_sr, target_sr=expected_sr, res_type='soxr_hq')
                    if resampled_audio.ndim == 1 and audio_arr_native_sr.ndim == 2: resampled_audio = resampled_audio[np.newaxis,:]
                    if resampled_audio.shape[0] == 1: resampled_audio = np.repeat(resampled_audio, 2, axis=0)
                    current_audio_processed = resampled_audio
                if current_audio_processed.shape[-1] > 0:
                    loaded_audios_at_expected_sr.append(current_audio_processed)
                    min_len_at_expected_sr = min(min_len_at_expected_sr, current_audio_processed.shape[-1])
            except Exception: pass # Suppress log for brevity, main dataset loader handles warnings
        if not loaded_audios_at_expected_sr or min_len_at_expected_sr == float('inf'): return None
        aligned_and_summed = None
        for s_arr in loaded_audios_at_expected_sr:
            trimmed_arr = s_arr[..., :min_len_at_expected_sr]
            aligned_and_summed = trimmed_arr if aligned_and_summed is None else aligned_and_summed + trimmed_arr
        return aligned_and_summed

    def stem_mixture(self, stem_name):
        track_types_for_stem = self.sources.get(stem_name, {})
        if not track_types_for_stem: return None
        all_track_type_mixtures = []
        min_len_across_track_types = float('inf')
        for track_type, component_paths_list in track_types_for_stem.items():
            mixed_components_for_track_type = self._load_and_mix_components(component_paths_list, self.sr)
            if mixed_components_for_track_type is not None:
                all_track_type_mixtures.append(mixed_components_for_track_type)
                min_len_across_track_types = min(min_len_across_track_types, mixed_components_for_track_type.shape[-1])
        if not all_track_type_mixtures or min_len_across_track_types == float('inf'): return None
        final_stem_audio = None
        for track_type_mix in all_track_type_mixtures:
            trimmed_track_type_mix = track_type_mix[..., :min_len_across_track_types]
            final_stem_audio = trimmed_track_type_mix if final_stem_audio is None else final_stem_audio + trimmed_track_type_mix
        return final_stem_audio

    @property
    def stems(self):
        stems_dict = {}
        for stem_name_default in all_stems:
            stem_audio = self.stem_mixture(stem_name_default)
            if stem_audio is not None: stems_dict[stem_name_default] = stem_audio
        return stems_dict

    @property
    def audio(self):
        current_stems = self.stems
        if 'mixture' in current_stems and current_stems['mixture'] is not None:
            mix_candidate = current_stems['mixture']
            if mix_candidate.ndim == 1: mix_candidate = np.stack((mix_candidate, mix_candidate), axis=0)
            elif mix_candidate.shape[0] == 1: mix_candidate = np.repeat(mix_candidate, 2, axis=0)
            elif mix_candidate.shape[0] > 2: mix_candidate = mix_candidate[:2,:]
            return mix_candidate.T if mix_candidate.shape[0] == 2 else None
        stems_to_sum = [current_stems[s_name] for s_name in ['vocals', 'drums', 'bass', 'other'] if s_name in current_stems and current_stems[s_name] is not None]
        if not stems_to_sum: return None
        mixed_audio_ch_first = pad_and_mix(stems_to_sum)
        return mixed_audio_ch_first.T if mixed_audio_ch_first is not None else None
    
    @property
    def activity(self):
        loaded_stems = self.stems; activity_dict = {}
        for stem_name, audio_data_ch_first in loaded_stems.items():
            activity_dict[stem_name] = None
            if audio_data_ch_first is not None and audio_data_ch_first.size > 0:
                try: activity_dict[stem_name] = compute_activity_signal(audio_data_ch_first[None, ...])[0]
                except Exception: pass
        return activity_dict

    def save_stems(self, path):
        os.makedirs(path, exist_ok=True)
        for stem_name, audio_data_ch_first in self.stems.items():
            if audio_data_ch_first is not None:
                save_audio(os.path.join(path, f"{stem_name}.wav"), audio_data_ch_first.T, sr=self.sr)

def pad_to_len(source_array_ch_first, length_to_pad_samples):
    if length_to_pad_samples > 0: return np.pad(source_array_ch_first, ((0, 0), (0, length_to_pad_samples)), mode='constant')
    return source_array_ch_first

def _standardize_to_stereo_and_filter(sources_list_ch_first):
    valid_stereo_sources = []
    for s_arr in sources_list_ch_first:
        if s_arr is None or not isinstance(s_arr, np.ndarray) or s_arr.ndim != 2 or s_arr.shape[0] == 0 or s_arr.shape[1] == 0: continue
        if s_arr.shape[0] == 1: valid_stereo_sources.append(np.repeat(s_arr, 2, axis=0))
        elif s_arr.shape[0] == 2: valid_stereo_sources.append(s_arr)
        elif s_arr.shape[0] > 2: valid_stereo_sources.append(s_arr[:2, :])
    return valid_stereo_sources

def pad_and_mix(sources_list_ch_first):
    processed_stereo_sources = _standardize_to_stereo_and_filter(sources_list_ch_first)
    if not processed_stereo_sources: return None
    max_len_samples = max(s.shape[1] for s in processed_stereo_sources)
    padded_sources = [pad_to_len(s_arr, max_len_samples - s_arr.shape[1]) for s_arr in processed_stereo_sources]
    return np.sum(np.stack(padded_sources, axis=0), axis=0) if padded_sources else None

def trim_and_mix(sources_list_ch_first):
    processed_stereo_sources = _standardize_to_stereo_and_filter(sources_list_ch_first)
    if not processed_stereo_sources: return None
    min_len_samples = min(s.shape[1] for s in processed_stereo_sources)
    trimmed_sources = [s[:, :min_len_samples] for s in processed_stereo_sources]
    return np.sum(np.stack(trimmed_sources, axis=0), axis=0) if trimmed_sources else None