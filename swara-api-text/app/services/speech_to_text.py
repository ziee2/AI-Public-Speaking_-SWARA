"""
Speech to Text Service
Wrapper untuk Whisper STT
"""

import whisper
import torch
import warnings
import os
from typing import Dict
from app.core.device import get_device, optimize_for_device
warnings.filterwarnings('ignore')


class SpeechToTextService:
    """Speech-to-Text service using Whisper"""
    
    def __init__(self, model_name: str = "medium", device: str = None, language: str = "id"):
        """Initialize Whisper model"""
        print(f"🎙️ Initializing Speech-to-Text service")
        print(f"📦 Loading Whisper model: {model_name}")
        
        # Auto-detect device if not specified
        if device is None or device == "auto":
            self.device = get_device()
            optimize_for_device(self.device)
        else:
            self.device = device
            print(f"💻 Using device: {self.device}")
        
        # Check if model is already cached
        # Use /data/.cache for Whisper (persistent storage on HF Pro)
        cache_dir = os.environ.get('WHISPER_CACHE', '/data/.cache')
        model_cache_path = os.path.join(cache_dir, f'{model_name}.pt')
        
        # Load Whisper model
        try:
            if os.path.exists(model_cache_path):
                print(f"✅ Loading from cache (pre-downloaded during build)")
            else:
                print(f"📥 Model not in cache, downloading '{model_name}'...")
                print(f"   This may take 1-2 minutes...")
            
            self.model = whisper.load_model(model_name, device=self.device, download_root=cache_dir)
            print("✅ Whisper model ready!\n")
        except Exception as e:
            print(f"❌ Failed to load model '{model_name}': {e}")
            print("⚙️ Falling back to 'base' model...")
            
            base_cache_path = os.path.join(cache_dir, 'base.pt')
            if os.path.exists(base_cache_path):
                print(f"✅ Loading base model from cache")
            else:
                print(f"📥 Downloading base model...")
            
            self.model = whisper.load_model("base", device=self.device, download_root=cache_dir)
            print("✅ Base model ready!\n")
        
        self.language = language
    
    def transcribe(self, audio_path: str, **kwargs) -> Dict:
        """
        Transcribe audio file to text
        
        Args:
            audio_path: Path ke file audio
            **kwargs: Additional Whisper parameters
            
        Returns:
            Dict: {'text': str, 'segments': list, 'language': str}
        """
        print(f"🎧 Transcribing: {audio_path}")
        
        try:
            # Try with word_timestamps first
            # Use FP16 for GPU to reduce memory and improve speed
            fp16 = self.device == "cuda"
            
            result = self.model.transcribe(
                audio_path,
                language=self.language,
                task="transcribe",
                word_timestamps=True,
                condition_on_previous_text=False,
                fp16=fp16,
                **kwargs
            )
        except Exception as e:
            print(f"⚠️ Transcription with word_timestamps failed: {e}")
            print(f"🔄 Retrying without word_timestamps...")
            
            # Fallback: transcribe without word_timestamps
            fp16 = self.device == "cuda"
            
            result = self.model.transcribe(
                audio_path,
                language=self.language,
                task="transcribe",
                condition_on_previous_text=False,
                fp16=fp16,
                **kwargs
            )
        
        print("✅ Transcription complete!\n")
        
        return {
            'text': result['text'],
            'segments': result.get('segments', []),
            'language': result.get('language', self.language)
        }
