"""
Filler Words Detection Service
Deteksi kata-kata pengisi (ehm, anu, itu, dll)
"""

import re
from typing import Dict, List


class FillerWordsService:
    """Service untuk deteksi kata pengisi"""
    
    # Daftar kata pengisi bahasa Indonesia
    FILLER_WORDS = [
        # Suara pengisi
        'eh', 'ehm', 'em', 'aa', 'ah', 'mm', 'hmm', 'uhh', 'umm',
        
        # Kata pengisi umum
        'anu', 'ini', 'itu', 'gitu', 'kayak', 'seperti',
        
        # Kata ragu
        'ya', 'kan', 'sih', 'deh', 'lah',
        
        # Kata repetitif
        'jadi', 'terus', 'nah', 'yaudah', 'gimana'
    ]
    
    def __init__(self):
        """Initialize service"""
        print("🗣️ Initializing Filler Words Service")
        print(f"📝 Monitoring {len(self.FILLER_WORDS)} filler words")
        print("✅ Filler Words Service ready!\n")
    
    def detect(self, transcript: str) -> Dict:
        """
        Deteksi kata pengisi dalam transkrip
        
        Args:
            transcript: Text transkrip
            
        Returns:
            Dict hasil deteksi
        """
        print("🔍 Detecting filler words...")
        
        if not transcript or not transcript.strip():
            return {
                'has_filler': False,
                'count': 0,
                'ratio': 0.0,
                'words_found': [],
                'total_words': 0,
                'positions': []
            }
        
        # Clean and split transcript
        words = transcript.lower().split()
        total_words = len(words)
        
        # Detect filler words
        filler_found = []
        filler_positions = []
        filler_count = 0
        
        for i, word in enumerate(words):
            # Clean word (remove punctuation)
            clean_word = re.sub(r'[^\w\s]', '', word)
            
            if clean_word in self.FILLER_WORDS:
                filler_count += 1
                filler_found.append(clean_word)
                filler_positions.append({
                    'word': clean_word,
                    'position': i,
                    'context': ' '.join(words[max(0, i-2):min(len(words), i+3)])
                })
        
        # Calculate ratio
        filler_ratio = filler_count / total_words if total_words > 0 else 0
        
        # Has filler?
        has_filler = filler_count > 0
        
        print(f"✅ Found {filler_count} filler words\n")
        
        return {
            'has_filler': has_filler,
            'count': filler_count,
            'ratio': round(filler_ratio, 3),
            'words_found': list(set(filler_found)),  # Unique words
            'total_words': total_words,
            'positions': filler_positions[:5]  # Return max 5 examples
        }
