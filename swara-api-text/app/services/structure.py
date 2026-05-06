"""
Structure Analysis Service
Analisis struktur berbicara (opening, content, closing)
"""

import pandas as pd
import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict
from app.core.device import get_device


class StructureService:
    """Analisis struktur public speaking"""
    
    def __init__(self, model_path: str = 'Cyberlace/swara-structure-model'):
        """
        Initialize model from Hugging Face Hub
        
        Args:
            model_path: HF Hub model name or local path
        """
        print("📊 Initializing Structure Service...")
        print(f"📦 Loading model from: {model_path}")
        
        # Auto-detect device
        self.device = get_device()
        
        # Load from Hugging Face Hub (with caching)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir="/.cache"
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            cache_dir="/.cache"
        )
        self.model.to(self.device)  # Move model to device
        self.model.eval()
        
        self.label_map = {0: 'opening', 1: 'content', 2: 'closing'}
        
        print("✅ Structure Service ready!\n")
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text menjadi kalimat-kalimat"""
        sentences = re.split(r'[.!?,;\n]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def predict_sentences(self, sentences: List[str], confidence_threshold: float = 0.7) -> List[Dict]:
        """Prediksi label untuk list kalimat"""
        results = []
        
        for idx, sentence in enumerate(sentences):
            inputs = self.tokenizer(
                sentence,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][predicted_class].item()
            
            predicted_label = self.label_map[predicted_class]
            
            # Jika opening/closing tapi confidence rendah → ubah jadi content
            if predicted_label in ['opening', 'closing'] and confidence < confidence_threshold:
                predicted_label = 'content'
            
            results.append({
                'sentence_idx': idx,
                'text': sentence,
                'predicted_label': predicted_label,
                'confidence': confidence
            })
        
        return results
    
    def apply_structure_rules(self, predictions: List[Dict]) -> List[Dict]:
        """Terapkan rules untuk memperbaiki struktur"""
        if not predictions:
            return predictions
        
        n = len(predictions)
        
        # Define keywords first (will be used in rules)
        closing_keywords = [
            'demikian', 'terima kasih', 'terimakasih', 'sekian', 'akhir kata',
            'wassalamualaikum', 'wassalam', 'waalaikumsalam',
            'sampai jumpa', 'sampai bertemu', 'salam penutup'
        ]
        
        opening_keywords = [
            'selamat pagi', 'selamat siang', 'selamat sore', 'selamat malam',
            'assalamualaikum', 'assalamu alaikum',
            'hadirin yang', 'bapak ibu', 'pertama-tama', 'izinkan saya',
            'perkenalkan', 'yang terhormat'
        ]
        
        # Rule 1: 2 kalimat pertama cenderung opening (HANYA jika ada opening keyword)
        for i in range(min(2, n)):
            text_lower = predictions[i]['text'].lower()
            has_opening_kw = any(kw in text_lower for kw in opening_keywords)
            
            if has_opening_kw and predictions[i]['confidence'] > 0.5:
                predictions[i]['predicted_label'] = 'opening'
        
        # Rule 2: 2 kalimat terakhir cenderung closing (HANYA jika ada closing keyword)
        for i in range(max(0, n-2), n):
            text_lower = predictions[i]['text'].lower()
            has_closing_kw = any(kw in text_lower for kw in closing_keywords)
            
            if has_closing_kw and predictions[i]['confidence'] > 0.5:
                predictions[i]['predicted_label'] = 'closing'
        
        # Rule 3: Keyword detection untuk semua kalimat (override model prediction)
        for pred in predictions:
            text_lower = pred['text'].lower()
            
            # Check OPENING first (lebih prioritas untuk kalimat awal)
            is_opening_keyword = any(kw in text_lower for kw in opening_keywords)
            
            # Check CLOSING - tapi EXCLUDE jika ada opening keyword
            # Ini prevent "assalamualaikum" salah dideteksi sebagai closing karena "salam"
            is_closing_keyword = any(kw in text_lower for kw in closing_keywords)
            
            if is_opening_keyword and not is_closing_keyword:
                pred['predicted_label'] = 'opening'
            elif is_closing_keyword and not is_opening_keyword:
                pred['predicted_label'] = 'closing'
        
        return predictions
    
    def segment_speech_structure(self, predictions: List[Dict]) -> Dict:
        """Grouping kalimat berdasarkan struktur"""
        structure = {
            'opening': [],
            'content': [],
            'closing': []
        }
        
        for pred in predictions:
            label = pred['predicted_label']
            structure[label].append(pred)
        
        return structure
    
    def calculate_score(self, structure: Dict) -> Dict:
        """Hitung skor berdasarkan struktur"""
        has_opening = len(structure['opening']) > 0
        has_content = len(structure['content']) > 0
        has_closing = len(structure['closing']) > 0
        
        if has_opening and has_content and has_closing:
            score = 5
            description = "Sempurna! Struktur lengkap (Pembuka, Isi, Penutup)"
        elif has_opening and has_content and not has_closing:
            score = 4
            description = "Baik. Ada pembuka dan isi, tapi kurang penutup"
        elif has_opening and not has_content and has_closing:
            score = 3
            description = "Cukup. Ada pembuka dan penutup, tapi isi kurang jelas"
        elif not has_opening and has_content and has_closing:
            score = 2
            description = "Perlu perbaikan. Kurang pembuka yang jelas"
        elif has_opening and not has_content and not has_closing:
            score = 1
            description = "Kurang lengkap. Hanya ada pembuka"
        else:
            score = 0
            description = "Struktur tidak terdeteksi dengan baik"
        
        return {
            'score': score,
            'max_score': 5,
            'description': description,
            'category': description.split('.')[0] if '.' in description else description,
            'has_opening': has_opening,
            'has_content': has_content,
            'has_closing': has_closing,
            'opening_count': len(structure['opening']),
            'content_count': len(structure['content']),
            'closing_count': len(structure['closing'])
        }
    
    def analyze(self, transcript: str, apply_rules: bool = True) -> Dict:
        """
        Analisis struktur speech
        
        Args:
            transcript: Teks lengkap dari speech
            apply_rules: Gunakan heuristic rules
            
        Returns:
            Dict berisi hasil analisis
        """
        print(f"📊 Analyzing structure...")
        
        # Split into sentences
        sentences = self.split_into_sentences(transcript)
        
        # Predict
        predictions = self.predict_sentences(sentences)
        
        # Apply rules
        if apply_rules:
            predictions = self.apply_structure_rules(predictions)
        
        # Segment structure
        structure = self.segment_speech_structure(predictions)
        
        # Calculate score
        score_result = self.calculate_score(structure)
        
        print("✅ Structure analysis complete!\n")
        
        return {
            'score': score_result['score'],
            'category': score_result['category'],
            'description': score_result['description'],
            'has_opening': score_result['has_opening'],
            'has_content': score_result['has_content'],
            'has_closing': score_result['has_closing'],
            'opening_count': score_result['opening_count'],
            'content_count': score_result['content_count'],
            'closing_count': score_result['closing_count'],
            'total_sentences': len(sentences)
        }
