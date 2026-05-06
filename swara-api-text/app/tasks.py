"""
Background tasks untuk RQ worker
"""

from typing import List, Optional
from app.services.audio_processor import AudioProcessor
from app.core.storage import delete_file


def process_audio_task(
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
):
    """
    Background task untuk process audio
    
    This function will be executed by RQ worker
    
    Args:
        audio_path: Path ke file audio
        reference_text: Teks referensi untuk artikulasi
        topic_id: ID topik dari database (Level 1-2)
        custom_topic: Topik custom dari user (Level 3)
        custom_keywords: List kata kunci dari GPT (Level 3)
        analyze_tempo: Flag tempo analysis
        analyze_articulation: Flag articulation analysis
        analyze_structure: Flag structure analysis
        analyze_keywords: Flag keyword analysis
        analyze_profanity: Flag profanity detection
    """
    import os
    task_id = os.path.basename(audio_path).split('.')[0]
    
    print(f"\n🔧 [WORKER] Processing started - Task: {task_id}")
    print(f"   File: {audio_path}")
    
    try:
        processor = AudioProcessor()
        
        result = processor.process_audio(
            audio_path=audio_path,
            reference_text=reference_text,
            topic_id=topic_id,
            custom_topic=custom_topic,
            custom_keywords=custom_keywords,
            analyze_tempo=analyze_tempo,
            analyze_articulation=analyze_articulation,
            analyze_structure=analyze_structure,
            analyze_keywords=analyze_keywords,
            analyze_profanity=analyze_profanity
        )
        
        # Cleanup file after processing
        try:
            delete_file(audio_path)
        except Exception as e:
            print(f"Warning: Could not delete file {audio_path}: {e}")
        
        print(f"✅ [WORKER] Processing completed - Task: {task_id}")
        
        return {
            'status': 'completed',
            'result': result
        }
        
    except Exception as e:
        print(f"❌ [WORKER] Processing failed - Task: {task_id}")
        print(f"   Error: {str(e)}")
        
        # Cleanup file on error
        try:
            delete_file(audio_path)
        except:
            pass
        
        return {
            'status': 'failed',
            'error': str(e)
        }
