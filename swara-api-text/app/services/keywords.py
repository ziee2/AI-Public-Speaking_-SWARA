"""
Keyword Relevance Service
Analisis relevansi kata kunci dengan topik menggunakan BERT embeddings
"""

import json
import re
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from app.core.device import get_device
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("⚠️  Warning: sentence-transformers not installed. Using fallback mode.")


class KeywordService:
    """Analisis relevansi kata kunci"""
    
    def __init__(self, dataset_path: str, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        """
        Initialize analyzer
        
        Args:
            dataset_path: Path ke file JSON dataset kata kunci
            model_name: Nama model Sentence Transformer
        """
        print("🔍 Initializing Keyword Service...")
        
        self.dataset_path = dataset_path
        self.topics = {}
        
        # Load dataset
        self.load_dataset(dataset_path)
        
        # Load BERT model
        if EMBEDDINGS_AVAILABLE:
            print(f"📦 Loading BERT model: {model_name}...")
            device = get_device()
            # Use HF_HOME cache directory (set in Dockerfile)
            import os
            cache_dir = os.environ.get('HF_HOME', '/.cache')
            self.model = SentenceTransformer(model_name, device=device, cache_folder=cache_dir)
            print("✅ Model loaded!")
        else:
            self.model = None
            print("⚠️  Running in fallback mode (no embeddings)")
        
        # Precompute embeddings
        self.keyword_embeddings = {}
        if self.model:
            self._precompute_embeddings()
        
        print("✅ Keyword Service ready!\n")
    
    def load_dataset(self, json_path: str):
        """Load dataset dari file JSON"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                self.topics = json.load(f)
            print(f"✅ Dataset loaded: {len(self.topics)} topics")
        except FileNotFoundError:
            raise FileNotFoundError(f"❌ Dataset file not found: {json_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"❌ Invalid JSON format: {e}")
    
    def _precompute_embeddings(self):
        """Precompute embeddings untuk semua keywords"""
        print("🔄 Precomputing embeddings...")
        
        for topic_id, topic_data in self.topics.items():
            self.keyword_embeddings[topic_id] = {}
            
            # Embed keywords
            keywords = topic_data['keywords']
            self.keyword_embeddings[topic_id]['keywords'] = self.model.encode(keywords)
            
            # Embed variants
            all_variants = []
            variant_mapping = []
            for keyword in keywords:
                variants = topic_data['variants'].get(keyword, [])
                for variant in variants:
                    all_variants.append(variant)
                    variant_mapping.append(keyword)
            
            if all_variants:
                self.keyword_embeddings[topic_id]['variants'] = {
                    'embeddings': self.model.encode(all_variants),
                    'mapping': variant_mapping,
                    'texts': all_variants
                }
        
        print("✅ Embeddings ready!")
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract sentences dari text"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def semantic_keyword_detection(self, text: str, topic_id: str, threshold: float = 0.5) -> Dict:
        """Deteksi keyword menggunakan semantic similarity"""
        if not self.model or topic_id not in self.keyword_embeddings:
            return self._fallback_detection(text, topic_id)
        
        sentences = self.extract_sentences(text)
        sentence_embeddings = self.model.encode(sentences)
        
        topic_data = self.topics[topic_id]
        keyword_embs = self.keyword_embeddings[topic_id]
        
        detected_keywords = defaultdict(list)
        
        # Direct keyword matching
        keyword_similarities = cosine_similarity(
            sentence_embeddings,
            keyword_embs['keywords']
        )
        
        for sent_idx, sentence in enumerate(sentences):
            for kw_idx, keyword in enumerate(topic_data['keywords']):
                similarity = keyword_similarities[sent_idx][kw_idx]
                
                if similarity >= threshold:
                    detected_keywords[keyword].append({
                        'type': 'semantic',
                        'sentence': sentence,
                        'similarity': float(similarity)
                    })
        
        # Variant matching
        if 'variants' in keyword_embs:
            variant_similarities = cosine_similarity(
                sentence_embeddings,
                keyword_embs['variants']['embeddings']
            )
            
            for sent_idx, sentence in enumerate(sentences):
                for var_idx, (variant, mapped_kw) in enumerate(
                    zip(keyword_embs['variants']['texts'],
                        keyword_embs['variants']['mapping'])
                ):
                    similarity = variant_similarities[sent_idx][var_idx]
                    
                    if similarity >= threshold:
                        if not any(d['type'] == 'variant' and d.get('variant') == variant
                                 for d in detected_keywords[mapped_kw]):
                            detected_keywords[mapped_kw].append({
                                'type': 'variant',
                                'variant': variant,
                                'sentence': sentence,
                                'similarity': float(similarity)
                            })
        
        # Exact string matching
        text_lower = text.lower()
        for keyword in topic_data['keywords']:
            if keyword in text_lower:
                if not any(d['type'] == 'exact' for d in detected_keywords[keyword]):
                    detected_keywords[keyword].insert(0, {
                        'type': 'exact',
                        'keyword': keyword,
                        'similarity': 1.0
                    })
            
            # Check variants
            for variant in topic_data['variants'].get(keyword, []):
                if variant.lower() in text_lower:
                    if not any(d['type'] == 'exact_variant' and d.get('variant') == variant
                             for d in detected_keywords[keyword]):
                        detected_keywords[keyword].insert(0, {
                            'type': 'exact_variant',
                            'variant': variant,
                            'similarity': 1.0
                        })
        
        return dict(detected_keywords)
    
    def _fallback_detection(self, text: str, topic_id: str) -> Dict:
        """Fallback method tanpa embeddings"""
        text_lower = text.lower()
        topic_data = self.topics[topic_id]
        detected_keywords = {}
        
        for keyword in topic_data['keywords']:
            detections = []
            
            if keyword in text_lower:
                detections.append({
                    'type': 'exact',
                    'keyword': keyword,
                    'similarity': 1.0
                })
            
            for variant in topic_data['variants'].get(keyword, []):
                if variant.lower() in text_lower:
                    detections.append({
                        'type': 'variant',
                        'variant': variant,
                        'similarity': 0.9
                    })
            
            if detections:
                detected_keywords[keyword] = detections
        
        return detected_keywords
    
    def calculate_score(self, detected_count: int) -> Dict:
        """Calculate skor berdasarkan jumlah keyword terdeteksi"""
        if detected_count >= 9:
            return {
                'score': 5,
                'category': 'Sangat Baik',
                'description': 'Coverage keyword sangat lengkap'
            }
        elif detected_count >= 7:
            return {
                'score': 4,
                'category': 'Baik',
                'description': 'Coverage keyword baik'
            }
        elif detected_count >= 5:
            return {
                'score': 3,
                'category': 'Cukup',
                'description': 'Coverage keyword cukup'
            }
        elif detected_count >= 3:
            return {
                'score': 2,
                'category': 'Buruk',
                'description': 'Coverage keyword kurang'
            }
        else:
            return {
                'score': 1,
                'category': 'Perlu Ditingkatkan',
                'description': 'Coverage keyword sangat rendah'
            }
    
    def analyze(
        self, 
        speech_text: str, 
        topic_id: str = None,
        custom_topic: str = None,
        custom_keywords: List[str] = None,
        threshold: float = 0.5
    ) -> Dict:
        """
        Analisis relevansi speech dengan topik
        
        Args:
            speech_text: Teks speech hasil transcription
            topic_id: ID topik dari database (untuk level 1-2)
            custom_topic: Topik custom dari user (untuk level 3)
            custom_keywords: List kata kunci dari GPT (untuk level 3)
            threshold: Similarity threshold
            
        Returns:
            Dict berisi hasil analisis
        """
        # Mode 1: Custom topic & keywords (Level 3 - dari GPT)
        if custom_topic and custom_keywords:
            print(f"🔍 Analyzing custom keywords for topic: {custom_topic}...")
            return self._analyze_custom_keywords(
                speech_text, 
                custom_topic, 
                custom_keywords,
                threshold
            )
        
        # Mode 2: Predefined topic (Level 1-2 - dari database)
        if topic_id:
            print(f"🔍 Analyzing keywords for topic {topic_id}...")
            
            if topic_id not in self.topics:
                return {"error": f"Topik '{topic_id}' tidak ditemukan"}
            
            topic_data = self.topics[topic_id]
            
            # Deteksi keywords
            detected_keywords = self.semantic_keyword_detection(
                speech_text, topic_id, threshold
            )
            
            missing_keywords = [
                kw for kw in topic_data['keywords']
                if kw not in detected_keywords
            ]
            
            # Calculate scores
            total_keywords = len(topic_data['keywords'])
            detected_count = len(detected_keywords)
            coverage_percentage = (detected_count / total_keywords) * 100
            
            score_result = self.calculate_score(detected_count)
            
            print(f"✅ Keyword analysis complete!\n")
            
            return {
                'score': score_result['score'],
                'category': score_result['category'],
                'description': score_result['description'],
                'topic_id': topic_id,
                'topic_title': topic_data['title'],
                'detected_count': detected_count,
                'total_keywords': total_keywords,
                'coverage_percentage': round(coverage_percentage, 1),
                'detected_keywords': list(detected_keywords.keys()),
                'missing_keywords': missing_keywords
            }
        
        # Mode 3: Error - tidak ada input
        return {"error": "Harus menyediakan topic_id ATAU (custom_topic + custom_keywords)"}
    
    def _analyze_custom_keywords(
        self,
        speech_text: str,
        custom_topic: str,
        custom_keywords: List[str],
        threshold: float = 0.5
    ) -> Dict:
        """
        Analisis dengan custom keywords dari GPT (untuk Level 3)
        
        Menghitung berapa kali setiap keyword disebutkan dalam speech
        """
        speech_lower = speech_text.lower()
        
        # Hitung kemunculan setiap keyword
        keyword_mentions = {}
        total_mentions = 0
        
        for keyword in custom_keywords:
            keyword_lower = keyword.lower()
            
            # Count exact matches (case-insensitive)
            count = speech_lower.count(keyword_lower)
            
            if count > 0:
                keyword_mentions[keyword] = {
                    'count': count,
                    'mentioned': True
                }
                total_mentions += count
            else:
                keyword_mentions[keyword] = {
                    'count': 0,
                    'mentioned': False
                }
        
        # Hitung statistik
        total_keywords = len(custom_keywords)
        mentioned_count = sum(1 for kw in keyword_mentions.values() if kw['mentioned'])
        not_mentioned = [kw for kw, data in keyword_mentions.items() if not data['mentioned']]
        coverage_percentage = (mentioned_count / total_keywords) * 100 if total_keywords > 0 else 0
        
        # Calculate score berdasarkan coverage
        score_result = self.calculate_score(mentioned_count)
        
        # Semantic analysis (optional - jika ada model)
        semantic_relevance = None
        if self.model:
            try:
                # Encode speech dan keywords
                speech_embedding = self.model.encode([speech_text])
                keywords_text = " ".join(custom_keywords)
                keywords_embedding = self.model.encode([keywords_text])
                
                # Calculate cosine similarity
                similarity = cosine_similarity(speech_embedding, keywords_embedding)[0][0]
                semantic_relevance = {
                    'similarity_score': round(float(similarity), 3),
                    'percentage': round(float(similarity) * 100, 1)
                }
            except Exception as e:
                print(f"⚠️  Semantic analysis failed: {e}")
        
        print(f"✅ Custom keyword analysis complete!\n")
        
        return {
            'score': score_result['score'],
            'category': score_result['category'],
            'description': score_result['description'],
            'mode': 'custom',
            'custom_topic': custom_topic,
            'total_keywords': total_keywords,
            'mentioned_count': mentioned_count,
            'total_mentions': total_mentions,
            'coverage_percentage': round(coverage_percentage, 1),
            'keyword_details': keyword_mentions,
            'not_mentioned': not_mentioned,
            'semantic_relevance': semantic_relevance
        }
