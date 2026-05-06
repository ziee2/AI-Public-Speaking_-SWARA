"""
Audio Processor - Main Orchestrator
Koordinasi semua analisis audio
"""

import time
from typing import Dict, Optional, List
from app.config import settings
from app.services.speech_to_text import SpeechToTextService
from app.services.tempo import TempoService
from app.services.articulation import ArticulationService, ProfanityDetector
from app.services.structure import StructureService
from app.services.keywords import KeywordService


class AudioProcessor:
    """Main orchestrator untuk audio analysis"""
    
    def __init__(self):
        """Initialize all services"""
        print("🚀 Initializing Audio Processor...")
        
        # Initialize services (lazy loading)
        self._stt_service = None
        self._tempo_service = None
        self._articulation_service = None
        self._structure_service = None
        self._keyword_service = None
        
        print("✅ Audio Processor ready!\n")
    
    @property
    def stt_service(self):
        """Lazy load STT service"""
        if self._stt_service is None:
            self._stt_service = SpeechToTextService(
                model_name=settings.WHISPER_MODEL,
                device="auto",  # Auto-detect GPU/CPU
                language="id"
            )
        return self._stt_service
    
    @property
    def tempo_service(self):
        """Lazy load Tempo service"""
        if self._tempo_service is None:
            self._tempo_service = TempoService()
        return self._tempo_service
    
    @property
    def articulation_service(self):
        """Lazy load Articulation service"""
        if self._articulation_service is None:
            self._articulation_service = ArticulationService()
        return self._articulation_service
    
    @property
    def structure_service(self):
        """Lazy load Structure service"""
        if self._structure_service is None:
            # Uses default 'Cyberlace/swara-structure-model' from HF Hub
            self._structure_service = StructureService()
        return self._structure_service
    
    @property
    def keyword_service(self):
        """Lazy load Keyword service"""
        if self._keyword_service is None:
            self._keyword_service = KeywordService(
                dataset_path=settings.KATA_KUNCI_PATH
            )
        return self._keyword_service
    
    def process_audio(
        self,
        audio_path: str,
        reference_text: Optional[str] = None,
        topic_id: Optional[str] = None,
        custom_topic: Optional[str] = None,
        custom_keywords: Optional[List[str]] = None,
        analyze_tempo: bool = True,
        analyze_articulation: bool = True,
        analyze_structure: bool = True,
        analyze_keywords: bool = False,
        analyze_profanity: bool = False
    ) -> Dict:
        """
        Process audio file dengan semua analisis yang diminta
        
        Args:
            audio_path: Path ke file audio
            reference_text: Teks referensi (untuk artikulasi)
            topic_id: ID topik dari database (untuk Level 1-2)
            custom_topic: Topik custom dari user (untuk Level 3)
            custom_keywords: List kata kunci dari GPT (untuk Level 3)
            analyze_tempo: Flag untuk analisis tempo
            analyze_articulation: Flag untuk analisis artikulasi
            analyze_structure: Flag untuk analisis struktur
            analyze_keywords: Flag untuk analisis kata kunci
            analyze_profanity: Flag untuk deteksi kata tidak senonoh
            
        Returns:
            Dict berisi semua hasil analisis
        """
        start_time = time.time()
        
        print("="*70)
        print("🎯 STARTING AUDIO ANALYSIS")
        print("="*70)
        print(f"📁 Audio file: {audio_path}")
        print(f"⚙️  Tempo: {analyze_tempo}")
        print(f"⚙️  Articulation: {analyze_articulation}")
        print(f"⚙️  Structure: {analyze_structure}")
        print(f"⚙️  Keywords: {analyze_keywords}")
        print(f"⚙️  Profanity: {analyze_profanity}")
        print("="*70 + "\n")
        
        results = {}
        
        # 1. Speech to Text (always required)
        print("📝 Step 1/6: Transcribing audio...")
        transcript_result = self.stt_service.transcribe(audio_path)
        transcript = transcript_result['text']
        results['transcript'] = transcript
        print(f"✅ Transcript: {transcript[:100]}...\n")
        
        # 2. Tempo Analysis
        if analyze_tempo:
            print("🎵 Step 2/6: Analyzing tempo...")
            results['tempo'] = self.tempo_service.analyze(audio_path, transcript)
            print(f"✅ Tempo score: {results['tempo']['score']}/5\n")
        
        # 3. Articulation Analysis
        if analyze_articulation:
            print("🗣️  Step 3/6: Analyzing articulation...")
            results['articulation'] = self.articulation_service.analyze(
                audio_path=audio_path,
                transcript=transcript,
                reference_text=reference_text if reference_text else None
            )
            print(f"✅ Articulation score: {results['articulation']['score']}/5\n")
        
        # 4. Structure Analysis
        if analyze_structure:
            print("📊 Step 4/6: Analyzing structure...")
            results['structure'] = self.structure_service.analyze(transcript)
            print(f"✅ Structure score: {results['structure']['score']}/5\n")
        
        # 5. Keyword Analysis
        if analyze_keywords:
            print("🔍 Step 5/6: Analyzing keywords...")
            
            # Custom keywords (Level 3 - dari GPT)
            if custom_topic and custom_keywords:
                results['keywords'] = self.keyword_service.analyze(
                    speech_text=transcript,
                    custom_topic=custom_topic,
                    custom_keywords=custom_keywords
                )
            # Predefined topic (Level 1-2 - dari database)
            elif topic_id:
                results['keywords'] = self.keyword_service.analyze(
                    speech_text=transcript,
                    topic_id=topic_id
                )
            else:
                print("⚠️  Step 5/6: Skipping keywords (no topic_id or custom_keywords)\n")
            
            if 'keywords' in results:
                print(f"✅ Keyword score: {results['keywords']['score']}/5\n")
        elif analyze_keywords:
            print("⚠️  Step 5/6: Keywords analysis disabled\n")
        
        # 6. Profanity Detection
        if analyze_profanity:
            print("🚫 Step 6/6: Detecting profanity...")
            results['profanity'] = ProfanityDetector.detect_profanity(transcript)
            status = "DETECTED" if results['profanity']['has_profanity'] else "CLEAN"
            print(f"✅ Profanity check: {status} ({results['profanity']['profanity_count']} words)\n")
        
        # Calculate overall score
        scores = []
        if 'tempo' in results:
            scores.append(results['tempo']['score'])
        if 'articulation' in results:
            scores.append(results['articulation']['score'])
        if 'structure' in results:
            scores.append(results['structure']['score'])
        if 'keywords' in results:
            scores.append(results['keywords']['score'])
        
        if scores:
            results['overall_score'] = round(sum(scores) / len(scores), 2)
        else:
            results['overall_score'] = 0
        
        processing_time = time.time() - start_time
        results['processing_time'] = round(processing_time, 2)
        
        print("="*70)
        print(f"✅ ANALYSIS COMPLETE")
        print(f"⏱️  Processing time: {processing_time:.2f}s")
        print(f"📊 Overall score: {results['overall_score']}/5")
        print("="*70 + "\n")
        
        return results
