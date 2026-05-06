"""
Unified Articulation Analysis Service
Gabungan PER-based (dengan reference) dan Clarity-based (tanpa reference)
"""

import torch
import torchaudio
import librosa
import numpy as np
from typing import Dict, List, Tuple, Optional
import re
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from rapidfuzz import fuzz


class ArticulationService:
    """Analisis artikulasi unified (dengan/tanpa reference text)"""
    
    def __init__(self):
        """Initialize Wav2Vec2 untuk phoneme recognition"""
        print("🗣️ Initializing Articulation Service...")
        
        # Load Wav2Vec2 Indonesian model untuk phoneme detection
        model_name = "indonesian-nlp/wav2vec2-indonesian-javanese-sundanese"
        
        # Set cache directory (production: /.cache, local: default)
        import os
        cache_dir = os.environ.get('HF_HOME', '/.cache')
        
        try:
            print(f"📦 Loading Wav2Vec2 model: {model_name}")
            print(f"📁 Cache directory: {cache_dir}")
            self.processor = Wav2Vec2Processor.from_pretrained(model_name, cache_dir=cache_dir)
            self.model = Wav2Vec2ForCTC.from_pretrained(model_name, cache_dir=cache_dir)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model_loaded = True
            print(f"💻 Device: {self.device}")
        except Exception as e:
            print(f"⚠️  Warning: Failed to load Wav2Vec2 model: {e}")
            print("⚠️  Will use fallback articulation analysis")
            self.model_loaded = False
        
        # Filler words bahasa Indonesia
        self.filler_words = [
            'eh', 'ehm', 'em', 'aa', 'ah', 'mm', 'hmm',
            'anu', 'itu', 'gitu', 'kayak', 'seperti',
            'ya', 'yaa', 'nah', 'terus', 'jadi', 'soalnya'
        ]
        
        print("✅ Articulation Service ready!\n")
    
    def extract_audio_features(self, audio_path: str) -> Tuple[Dict, torch.Tensor, int]:
        """Extract fitur audio untuk analisis artikulasi"""
        print(f"🎵 Extracting audio features from: {audio_path}")
        
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample ke 16kHz jika perlu
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
            sr = 16000
        
        # Convert to numpy
        audio = waveform.squeeze().numpy()
        
        # Extract features
        features = {
            'duration': len(audio) / sr,
            'rms_energy': np.sqrt(np.mean(audio**2)),
            'zero_crossing_rate': librosa.zero_crossings(audio).sum() / len(audio),
            'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)),
            'spectral_rolloff': np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
        }
        
        print(f"   Duration: {features['duration']:.2f}s")
        print(f"   RMS Energy: {features['rms_energy']:.4f}")
        
        return features, waveform, sr
    
    def analyze_phoneme_clarity(self, waveform: torch.Tensor, sr: int) -> Dict:
        """Analisis kejelasan phoneme menggunakan Wav2Vec2"""
        print("🔍 Analyzing phoneme clarity...")
        
        if self.model is None or self.processor is None:
            print("⚠️  Wav2Vec2 not available, using fallback")
            return {
                'clarity_score': 70.0,  # Default score
                'avg_confidence': 0.7,
                'min_confidence': 0.5,
                'confidence_std': 0.15,
                'consistency': 0.85
            }
        
        try:
            # Prepare input
            inputs = self.processor(
                waveform.squeeze().numpy(),
                sampling_rate=sr,
                return_tensors="pt",
                padding=True
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get logits
            with torch.no_grad():
                logits = self.model(**inputs).logits
            
            # Get confidence scores
            probs = torch.nn.functional.softmax(logits, dim=-1)
            max_probs = torch.max(probs, dim=-1).values
            
            # Calculate clarity metrics
            avg_confidence = torch.mean(max_probs).item()
            min_confidence = torch.min(max_probs).item()
            confidence_std = torch.std(max_probs).item()
            
            # Clarity score (0-100)
            clarity_score = avg_confidence * 100
            
            print(f"   Clarity Score: {clarity_score:.2f}%")
            print(f"   Avg Confidence: {avg_confidence:.3f}")
            
            return {
                'clarity_score': clarity_score,
                'avg_confidence': avg_confidence,
                'min_confidence': min_confidence,
                'confidence_std': confidence_std,
                'consistency': 1 - confidence_std
            }
        except Exception as e:
            print(f"⚠️  Error in phoneme clarity analysis: {e}")
            return {
                'clarity_score': 70.0,
                'avg_confidence': 0.7,
                'min_confidence': 0.5,
                'confidence_std': 0.15,
                'consistency': 0.85
            }
    
    def detect_filler_words(self, transcript: str) -> Dict:
        """Deteksi kata-kata pengisi (filler words)"""
        print("🔎 Detecting filler words...")
        
        # Split by whitespace to preserve original form
        words = transcript.split()
        total_words = len(words)
        
        if total_words == 0:
            return {
                'filler_count': 0,
                'filler_words_found': []
            }
        
        # Count filler words using fuzzy matching + exact match for short words
        filler_found = []
        filler_count = 0
        
        for word in words:
            # Clean word for checking (lowercase, remove punctuation)
            clean_word = re.sub(r'[^\w\s]', '', word.lower())
            
            # Skip empty words
            if not clean_word:
                continue
            
            is_filler = False
            
            # For short words (2-3 chars), use exact match to avoid false positives
            if len(clean_word) <= 3:
                if clean_word in self.filler_words:
                    is_filler = True
            else:
                # For longer words, use fuzzy matching with 90% threshold
                for filler_word in self.filler_words:
                    similarity = fuzz.ratio(clean_word, filler_word)
                    if similarity >= 90:  # 90% threshold untuk presisi lebih tinggi
                        is_filler = True
                        break
            
            if is_filler:
                filler_count += 1
                # Keep original word form (with punctuation like 'ehm...')
                if word not in filler_found:
                    filler_found.append(word)
        
        # Calculate filler ratio
        filler_ratio = filler_count / total_words if total_words > 0 else 0
        
        print(f"   Filler Words: {filler_count}/{total_words} ({filler_ratio*100:.1f}%)")
        if filler_found:
            print(f"   Found: {', '.join(filler_found)}")
        
        return {
            'filler_count': filler_count,
            'filler_ratio': filler_ratio,
            'filler_words_found': filler_found
        }
    
    def analyze_speech_rate_stability(self, audio_path: str) -> Dict:
        """Analisis kestabilan kecepatan bicara"""
        print("📊 Analyzing speech rate stability...")
        
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=16000)
            
            # Detect onsets (segment boundaries)
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='frames')
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)
            
            if len(onset_times) < 2:
                print("   ⚠️  Not enough onsets detected")
                return {
                    'stability_score': 50.0,
                    'avg_syllable_rate': 0,
                    'rate_std': 0
                }
            
            # Calculate inter-onset intervals (IOI)
            ioi = np.diff(onset_times)
            
            # Speech rate metrics
            avg_rate = 1 / np.mean(ioi) if len(ioi) > 0 else 0
            rate_std = np.std(ioi) if len(ioi) > 0 else 0
            
            # Stability score (semakin rendah std, semakin stabil)
            stability_score = max(0, 100 - (rate_std * 100))
            
            print(f"   Stability Score: {stability_score:.2f}%")
            print(f"   Syllable Rate: {avg_rate:.2f}/s")
            
            return {
                'stability_score': stability_score,
                'avg_syllable_rate': avg_rate,
                'rate_std': rate_std,
                'onset_count': len(onset_times)
            }
        except Exception as e:
            print(f"⚠️  Error in stability analysis: {e}")
            return {
                'stability_score': 60.0,
                'avg_syllable_rate': 0,
                'rate_std': 0
            }
    
    def calculate_per(self, reference: str, hypothesis: str) -> float:
        """
        Calculate Phoneme Error Rate (word-level approximation)
        Using Levenshtein distance
        """
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        m, n = len(ref_words), len(hyp_words)
        
        # Dynamic programming for edit distance
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_words[i-1] == hyp_words[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],    # deletion
                        dp[i][j-1],    # insertion
                        dp[i-1][j-1]   # substitution
                    )
        
        errors = dp[m][n]
        per = (errors / m * 100) if m > 0 else 0
        return per
    
    def calculate_overall_score(
        self,
        clarity: Dict,
        filler: Dict,
        stability: Dict,
        features: Dict,
        per: Optional[float] = None
    ) -> Dict:
        """Hitung skor keseluruhan artikulasi"""
        print("\n🎯 Calculating overall articulation score...")
        
        # Clarity score (0-100)
        clarity_score = clarity['clarity_score']
        
        # Filler score (0-100, semakin sedikit filler semakin baik)
        filler_score = max(0, 100 - (filler['filler_ratio'] * 200))
        
        # Stability score (0-100)
        stability_score = stability['stability_score']
        
        # Energy score (normalized RMS energy)
        energy_score = min(100, features['rms_energy'] * 1000)
        
        if per is not None:
            # Mode 1: WITH REFERENCE - PER based
            # Weight untuk dengan reference
            weights = {
                'per': 0.4,          # 40% - PER adalah gold standard
                'clarity': 0.3,      # 30% - Kejelasan phoneme
                'stability': 0.2,    # 20% - Kestabilan tempo
                'energy': 0.1        # 10% - Energi/volume bicara
            }
            
            # PER score (lower is better, invert to 0-100 scale)
            per_score = max(0, 100 - per)
            
            # Weighted average
            total_score = (
                per_score * weights['per'] +
                clarity_score * weights['clarity'] +
                stability_score * weights['stability'] +
                energy_score * weights['energy']
            )
            
            # Convert to 1-5 scale
            score_5 = int(np.clip(total_score / 20, 1, 5))
            
            # Category
            if score_5 >= 5:
                category = "Sempurna"
                reason = f"PER sangat rendah ({per:.1f}%), artikulasi sangat jelas"
            elif score_5 >= 4:
                category = "Baik"
                reason = f"PER rendah ({per:.1f}%), artikulasi jelas"
            elif score_5 >= 3:
                category = "Cukup"
                reason = f"PER sedang ({per:.1f}%), artikulasi cukup jelas"
            elif score_5 >= 2:
                category = "Kurang"
                reason = f"PER tinggi ({per:.1f}%), banyak kesalahan pengucapan"
            else:
                category = "Buruk"
                reason = f"PER sangat tinggi ({per:.1f}%), artikulasi tidak jelas"
            
            print(f"\n📊 Score Breakdown (WITH REFERENCE):")
            print(f"   PER:       {per:.1f}% → Score: {per_score:.1f}% (weight: {weights['per']*100:.0f}%)")
            print(f"   Clarity:   {clarity_score:.1f}% (weight: {weights['clarity']*100:.0f}%)")
            print(f"   Stability: {stability_score:.1f}% (weight: {weights['stability']*100:.0f}%)")
            print(f"   Energy:    {energy_score:.1f}% (weight: {weights['energy']*100:.0f}%)")
            print(f"   TOTAL:     {total_score:.1f}% → {score_5}/5")
            
            return {
                'score': score_5,
                'category': category,
                'reason': reason,
                'mode': 'with_reference',
                'details': {
                    'per': round(per, 2),
                    'per_score': round(per_score, 2),
                    'clarity_score': round(clarity_score, 2),
                    'stability_score': round(stability_score, 2),
                    'energy_score': round(energy_score, 2),
                    'total_score': round(total_score, 2)
                }
            }
        else:
            # Mode 2: WITHOUT REFERENCE - Clarity based
            # Weight untuk tanpa reference (TANPA filler component)
            weights = {
                'clarity': 0.5,      # 50% - Kejelasan phoneme paling penting
                'stability': 0.3,    # 30% - Kestabilan tempo
                'energy': 0.2        # 20% - Energi/volume bicara
            }
            
            # Weighted average
            total_score = (
                clarity_score * weights['clarity'] +
                stability_score * weights['stability'] +
                energy_score * weights['energy']
            )
            
            # Convert to 1-5 scale based on percentage ranges
            # 81-100% = 5, 61-80% = 4, 41-60% = 3, 21-40% = 2, 0-20% = 1
            if total_score >= 81:
                score_5 = 5
                category = "Sempurna"
                reason = f"Artikulasi sangat jelas ({total_score:.1f}%) dan konsisten"
            elif total_score >= 61:
                score_5 = 4
                category = "Baik"
                reason = f"Artikulasi jelas ({total_score:.1f}%) dengan tempo stabil"
            elif total_score >= 41:
                score_5 = 3
                category = "Cukup"
                reason = f"Artikulasi cukup jelas ({total_score:.1f}%), ada sedikit variasi tempo"
            elif total_score >= 21:
                score_5 = 2
                category = "Kurang"
                reason = f"Artikulasi kurang jelas ({total_score:.1f}%), tempo tidak stabil"
            else:
                score_5 = 1
                category = "Buruk"
                reason = f"Artikulasi tidak jelas ({total_score:.1f}%) dan sulit dipahami"
            
            print(f"\n📊 Score Breakdown (WITHOUT REFERENCE):")
            print(f"   Clarity:   {clarity_score:.1f}% (weight: {weights['clarity']*100:.0f}%)")
            print(f"   Stability: {stability_score:.1f}% (weight: {weights['stability']*100:.0f}%)")
            print(f"   Energy:    {energy_score:.1f}% (weight: {weights['energy']*100:.0f}%)")
            print(f"   TOTAL:     {total_score:.1f}% → {score_5}/5")
            
            return {
                'score': score_5,
                'category': category,
                'reason': reason,
                'mode': 'without_reference',
                'details': {
                    'clarity_score': round(clarity_score, 2),
                    'stability_score': round(stability_score, 2),
                    'energy_score': round(energy_score, 2),
                    'total_score': round(total_score, 2)
                }
            }
    
    def analyze(self, audio_path: str, transcript: str, reference_text: Optional[str] = None) -> Dict:
        """
        Analisis artikulasi unified (auto-detect mode)
        
        Args:
            audio_path: Path ke file audio
            transcript: Hasil transcription
            reference_text: Text reference (optional, jika ada gunakan PER mode)
            
        Returns:
            Dict hasil analisis artikulasi
        """
        print("\n" + "="*60)
        if reference_text and reference_text.strip():
            print("🗣️  ARTICULATION ANALYSIS (WITH REFERENCE)")
            mode_desc = "PER-based"
        else:
            print("🗣️  ARTICULATION ANALYSIS (WITHOUT REFERENCE)")
            mode_desc = "Clarity-based"
        print("="*60)
        
        # Extract audio features
        features, waveform, sr = self.extract_audio_features(audio_path)
        
        # Analyze phoneme clarity
        clarity = self.analyze_phoneme_clarity(waveform, sr)
        
        # Detect filler words
        filler = self.detect_filler_words(transcript)
        
        # Analyze speech rate stability
        stability = self.analyze_speech_rate_stability(audio_path)
        
        # Calculate PER if reference provided
        per = None
        if reference_text and reference_text.strip():
            print(f"\n📝 Calculating PER...")
            per = self.calculate_per(reference_text, transcript)
            print(f"   PER: {per:.2f}%")
        
        # Calculate overall score
        result = self.calculate_overall_score(clarity, filler, stability, features, per)
        
        # Add detailed metrics
        result['clarity_metrics'] = {
            'avg_confidence': round(clarity['avg_confidence'], 3),
            'consistency': round(clarity['consistency'], 3)
        }
        
        result['filler_count'] = filler['filler_count']
        result['filler_words'] = filler['filler_words_found']
        
        result['stability_metrics'] = {
            'syllable_rate': round(stability['avg_syllable_rate'], 2),
            'rate_variation': round(stability['rate_std'], 3)
        }
        
        if per is not None:
            result['metrics'] = {
                'reference_words': len(reference_text.split()),
                'transcript_words': len(transcript.split()),
                'per': round(per, 2)
            }
        
        print("\n✅ Articulation analysis complete!")
        print("="*60 + "\n")
        
        return result
\
\

class ProfanityDetector:
    """Deteksi kata tidak senonoh menggunakan hybrid approach (exact + fuzzy + pattern)"""
    
    # Base profanity words (kata dasar)
    PROFANITY_WORDS = {
        'anjir', 'anjay', 'njir', 'njay', 'anjrit', 'njrit', 'anjim', 'anjing', 
        'anjrot', 'asu', 'babi', 'bacot', 'bajingan', 'banci', 'bangke', 'bangor', 
        'bangsat', 'bego', 'bejad', 'bencong', 'bodat', 'bodoh', 'bugil', 'bundir', 
        'bunuh', 'burik', 'burit', 'cawek', 'cemen', 'cipok', 'cium', 'colai', 'coli', 
        'colmek', 'cukimai', 'cukimay', 'culun', 'cumbu', 'dancuk', 'dewasa', 'dick', 
        'dildo', 'encuk', 'fuck', 'gay', 'gei', 'gembel', 'gey', 'gigolo', 'gila', 
        'goblog', 'goblok', 'haram', 'hencet', 'hentai', 'idiot', 'jablai', 'jablay', 
        'jancok', 'jancuk', 'jangkik', 'jembut', 'jilat', 'jingan', 'kampang', 
        'keparat', 'kimak', 'kirik', 'klentit', 'klitoris', 'konthol', 'kontol', 
        'koplok', 'kunyuk', 'kutang', 'kutis', 'kwontol', 'lonte', 'maho', 
        'masturbasi', 'matane', 'mati', 'memek', 'mesum', 'modar', 'modyar', 'mokad', 
        'najis', 'nazi', 'ndhasmu', 'nenen', 'ngentot', 'ngolom', 'ngulum', 'nigga', 
        'nigger', 'onani', 'oon', 'orgasme', 'paksa', 'pantat', 'pantek', 'pecun', 
        'peli', 'penis', 'pentil', 'pepek', 'perek', 'perkosa', 'piatu', 'porno', 
        'pukimak', 'qontol', 'selangkang', 'sempak', 'senggama', 'setan', 'setubuh', 
        'shit', 'silet', 'silit', 'sinting', 'sodomi', 'stres', 'telanjang', 'telaso', 
        'tete', 'tewas', 'titit', 'togel', 'toket', 'tolol', 'tusbol', 'urin', 'vagina'
    }
    
    # Multi-word profanity phrases
    PROFANITY_PHRASES = {
        'gak ada otak', 'tidak ada otak', 'ga ada otak'
    }
    
    # Character substitution map (leet speak)
    CHAR_SUBSTITUTIONS = {
        '0': 'o', '1': 'i', '3': 'e', '4': 'a', '5': 's', 
        '7': 't', '8': 'b', '@': 'a', '$': 's', '*': ''
    }
    
    @classmethod
    def normalize_word(cls, word: str) -> str:
        """Normalize word by replacing common character substitutions"""
        normalized = word.lower()
        for char, replacement in cls.CHAR_SUBSTITUTIONS.items():
            normalized = normalized.replace(char, replacement)
        return normalized
    
    @classmethod
    def detect_profanity(cls, text: str) -> dict:
        """
        Detect profanity using hybrid approach:
        1. Exact match for quick detection
        2. Fuzzy match for typo variations
        3. Pattern matching for character substitution (leet speak)
        """
        text_lower = text.lower()
        
        # Extract words and normalize
        raw_words = re.findall(r'\w+', text_lower)
        
        found_profanity = []
        profanity_count = 0
        
        # Step 1: Check multi-word phrases first
        for phrase in cls.PROFANITY_PHRASES:
            if phrase in text_lower:
                profanity_count += 1
                if phrase not in found_profanity:
                    found_profanity.append(phrase)
        
        # Step 2: Check individual words
        for word in raw_words:
            is_profane = False
            matched_word = word
            
            # A. Exact match (fastest)
            if word in cls.PROFANITY_WORDS:
                is_profane = True
            
            # B. Normalize and check (handle leet speak: t0l0l → tolol)
            elif len(word) > 0:
                normalized = cls.normalize_word(word)
                if normalized in cls.PROFANITY_WORDS:
                    is_profane = True
                    matched_word = normalized
            
            # C. Fuzzy match for typo variations (anjiir, anjiirr, etc.)
            if not is_profane and len(word) > 3:
                for profane_word in cls.PROFANITY_WORDS:
                    # Only compare words with similar length (±3 chars)
                    if abs(len(word) - len(profane_word)) <= 3:
                        similarity = fuzz.ratio(word, profane_word)
                        if similarity >= 85:  # 85% threshold for profanity
                            is_profane = True
                            matched_word = profane_word
                            break
            
            if is_profane:
                profanity_count += 1
                # Keep original word if not already in list
                if word not in found_profanity:
                    found_profanity.append(word)
        
        return {
            'has_profanity': len(found_profanity) > 0,
            'profanity_count': profanity_count,
            'profanity_words': list(set(found_profanity))
        }
