"""
Tempo Analysis Service
Analisis tempo dan jeda bicara menggunakan Silero VAD
"""

import torch
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class TempoService:
    """Analisis tempo dan jeda bicara"""
    
    def __init__(self):
        """Initialize Silero VAD model"""
        print("🔄 Loading Silero VAD model...")
        torch.set_num_threads(1)
        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad', 
            model='silero_vad', 
            force_reload=False
        )
        (self.get_speech_timestamps, 
         self.save_audio, 
         self.read_audio, 
         self.VADIterator, 
         self.collect_chunks) = utils
        print("✅ Silero VAD model loaded!\n")
    
    def analyze(self, audio_path: str, transcription: str, sampling_rate: int = 16000) -> Dict:
        """
        Analisis tempo berdasarkan jumlah kata per menit dan deteksi jeda panjang
        
        Kriteria penilaian:
        - Poin 5 (Sangat Baik): 140-150 kata dalam 48-60 detik, tidak ada jeda >3 detik
        - Poin 4 (Baik): 110-139 kata dalam 36-60 detik, tidak ada jeda >3 detik
        - Poin 3 (Cukup): 60-109 kata dalam 60 detik, tidak ada jeda >3 detik
        - Poin 2 (Buruk): <60 kata dalam 60 detik, tidak ada jeda >3 detik
        - Poin 1 (Perlu Ditingkatkan): Berhenti sebelum 60 detik ATAU ada jeda >3 detik
        
        Args:
            audio_path: Path ke file audio
            transcription: Teks hasil transcription untuk hitung jumlah kata
            sampling_rate: Sample rate audio (default: 16000)
            
        Returns:
            Dict berisi hasil analisis lengkap
        """
        print(f"🎧 Analyzing tempo: {audio_path}")
        
        # Load audio
        wav = self.read_audio(audio_path)
        
        # Deteksi segmen bicara
        speech_timestamps = self.get_speech_timestamps(
            wav, self.model, sampling_rate=sampling_rate
        )
        
        # Hitung total durasi audio
        total_duration_sec = len(wav) / sampling_rate
        
        # Hitung jumlah kata dari transcription
        word_count = len(transcription.split())
        
        # Hitung kata per menit (normalize ke 60 detik)
        words_per_minute = (word_count / total_duration_sec) * 60 if total_duration_sec > 0 else 0
        
        # Deteksi jeda panjang (>3 detik)
        long_pauses = []
        has_long_pause = False
        
        data = []
        for i, seg in enumerate(speech_timestamps):
            start_time = seg['start'] / sampling_rate
            end_time = seg['end'] / sampling_rate
            duration = end_time - start_time
            
            if i == 0:
                pause_before = start_time
            else:
                pause_before = start_time - (speech_timestamps[i - 1]['end'] / sampling_rate)
            
            # Check jeda panjang
            if pause_before > 3.0:
                has_long_pause = True
                long_pauses.append({
                    'after_segment': i,
                    'pause_duration': round(pause_before, 2)
                })
            
            data.append({
                'segment': i + 1,
                'start_sec': round(start_time, 2),
                'end_sec': round(end_time, 2),
                'duration_sec': round(duration, 2),
                'pause_before_sec': round(pause_before, 2)
            })
        
        # Tentukan skor berdasarkan kriteria
        if total_duration_sec < 60 or has_long_pause:
            # Poin 1: Berhenti sebelum 60 detik ATAU ada jeda >3 detik
            poin = 1
            kategori = "Perlu Ditingkatkan"
            if total_duration_sec < 60:
                alasan = f"Durasi bicara hanya {round(total_duration_sec, 1)} detik (kurang dari 60 detik)"
            else:
                alasan = f"Terdapat {len(long_pauses)} jeda lebih dari 3 detik"
        elif words_per_minute >= 140 and words_per_minute <= 150 and total_duration_sec >= 48:
            # Poin 5: 140-150 kata dalam 48-60 detik
            poin = 5
            kategori = "Sangat Baik"
            alasan = f"Tempo ideal: {round(words_per_minute, 1)} kata/menit dalam {round(total_duration_sec, 1)} detik"
        elif words_per_minute >= 110 and words_per_minute <= 139 and total_duration_sec >= 36:
            # Poin 4: 110-139 kata dalam 36-60 detik
            poin = 4
            kategori = "Baik"
            alasan = f"Tempo baik: {round(words_per_minute, 1)} kata/menit dalam {round(total_duration_sec, 1)} detik"
        elif words_per_minute >= 60 and words_per_minute <= 109:
            # Poin 3: 60-109 kata dalam 60 detik
            poin = 3
            kategori = "Cukup"
            alasan = f"Tempo cukup: {round(words_per_minute, 1)} kata/menit"
        else:
            # Poin 2: <60 kata dalam 60 detik
            poin = 2
            kategori = "Buruk"
            alasan = f"Tempo lambat: hanya {round(words_per_minute, 1)} kata/menit"
        
        print("✅ Tempo analysis complete!\n")
        
        return {
            'score': poin,
            'category': kategori,
            'reason': alasan,
            'total_duration_sec': round(total_duration_sec, 2),
            'word_count': word_count,
            'words_per_minute': round(words_per_minute, 1),
            'has_long_pause': has_long_pause,
            'long_pauses': long_pauses,
            'total_segments': len(speech_timestamps),
            # 'segments': data
        }
