"""
Services module
"""

from app.services.speech_to_text import SpeechToTextService
from app.services.tempo import TempoService
from app.services.articulation import ArticulationService
from app.services.structure import StructureService
from app.services.keywords import KeywordService
from app.services.audio_processor import AudioProcessor

__all__ = [
    'SpeechToTextService',
    'TempoService',
    'ArticulationService',
    'StructureService',
    'KeywordService',
    'AudioProcessor'
]
