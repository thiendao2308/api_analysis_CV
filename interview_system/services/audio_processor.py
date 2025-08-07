import logging
import speech_recognition as sr
import tempfile
import os
from typing import Optional, Dict, List
import wave
import numpy as np

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Service for processing audio input and converting to text"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 4000
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        
        # Configure for better accuracy
        self.recognizer.phrase_threshold = 0.3
        self.recognizer.non_speaking_duration = 0.5
    
    def process_audio_file(self, audio_file_path: str) -> Dict:
        """Process audio file and convert to text"""
        
        try:
            logger.info(f"Processing audio file: {audio_file_path}")
            
            # Load audio file
            with sr.AudioFile(audio_file_path) as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Record audio
                audio = self.recognizer.record(source)
            
            # Convert to text
            text_result = self._convert_speech_to_text(audio)
            
            # Analyze audio quality
            quality_metrics = self._analyze_audio_quality(audio_file_path)
            
            result = {
                "success": True,
                "transcript": text_result.get("transcript", ""),
                "confidence": text_result.get("confidence", 0),
                "language": text_result.get("language", "en-US"),
                "duration": quality_metrics.get("duration", 0),
                "quality_score": quality_metrics.get("quality_score", 0),
                "audio_metrics": quality_metrics
            }
            
            logger.info(f"Audio processing completed. Confidence: {result['confidence']}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing audio file: {e}")
            return {
                "success": False,
                "error": str(e),
                "transcript": "",
                "confidence": 0
            }
    
    def process_audio_data(self, audio_data: bytes, format: str = "wav") -> Dict:
        """Process audio data from memory"""
        
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            # Process the temporary file
            result = self.process_audio_file(temp_file_path)
            
            # Clean up
            os.unlink(temp_file_path)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing audio data: {e}")
            return {
                "success": False,
                "error": str(e),
                "transcript": "",
                "confidence": 0
            }
    
    def _convert_speech_to_text(self, audio) -> Dict:
        """Convert speech to text using multiple engines"""
        
        # Try Google Speech Recognition first
        try:
            text = self.recognizer.recognize_google(audio)
            confidence = 0.8  # Google doesn't provide confidence scores
            return {
                "transcript": text,
                "confidence": confidence,
                "language": "en-US",
                "engine": "google"
            }
        except sr.UnknownValueError:
            logger.warning("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            logger.warning(f"Google Speech Recognition error: {e}")
        
        # Try Sphinx (offline) as fallback
        try:
            text = self.recognizer.recognize_sphinx(audio)
            return {
                "transcript": text,
                "confidence": 0.6,  # Sphinx confidence is lower
                "language": "en-US",
                "engine": "sphinx"
            }
        except sr.UnknownValueError:
            logger.warning("Sphinx could not understand audio")
        except Exception as e:
            logger.warning(f"Sphinx error: {e}")
        
        # Return empty result if all engines fail
        return {
            "transcript": "",
            "confidence": 0,
            "language": "en-US",
            "engine": "none"
        }
    
    def _analyze_audio_quality(self, audio_file_path: str) -> Dict:
        """Analyze audio quality metrics"""
        
        try:
            with wave.open(audio_file_path, 'rb') as audio_file:
                # Get audio properties
                frames = audio_file.getnframes()
                rate = audio_file.getframerate()
                duration = frames / float(rate)
                
                # Read audio data for analysis
                audio_data = audio_file.readframes(frames)
                
                # Convert to numpy array for analysis
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                
                # Calculate quality metrics
                rms = np.sqrt(np.mean(audio_array**2))
                peak = np.max(np.abs(audio_array))
                dynamic_range = 20 * np.log10(peak / (rms + 1e-10))
                
                # Quality score based on multiple factors
                quality_score = min(100, max(0, (
                    min(100, rms / 1000) * 0.3 +  # Volume level
                    min(100, dynamic_range) * 0.4 +  # Dynamic range
                    min(100, duration * 10) * 0.3   # Duration factor
                )))
                
                return {
                    "duration": duration,
                    "sample_rate": rate,
                    "channels": audio_file.getnchannels(),
                    "rms_level": rms,
                    "peak_level": peak,
                    "dynamic_range": dynamic_range,
                    "quality_score": quality_score
                }
                
        except Exception as e:
            logger.error(f"Error analyzing audio quality: {e}")
            return {
                "duration": 0,
                "quality_score": 0,
                "error": str(e)
            }
    
    def validate_audio_format(self, file_path: str) -> bool:
        """Validate if audio file format is supported"""
        
        supported_formats = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
        file_extension = os.path.splitext(file_path)[1].lower()
        
        return file_extension in supported_formats
    
    def get_audio_info(self, file_path: str) -> Dict:
        """Get basic audio file information"""
        
        try:
            with wave.open(file_path, 'rb') as audio_file:
                return {
                    "channels": audio_file.getnchannels(),
                    "sample_width": audio_file.getsampwidth(),
                    "frame_rate": audio_file.getframerate(),
                    "frames": audio_file.getnframes(),
                    "duration": audio_file.getnframes() / audio_file.getframerate(),
                    "format": "WAV"
                }
        except Exception as e:
            logger.error(f"Error getting audio info: {e}")
            return {"error": str(e)}
    
    def enhance_audio(self, audio_data: bytes) -> bytes:
        """Basic audio enhancement (noise reduction, normalization)"""
        
        try:
            # Convert to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Normalize audio
            max_val = np.max(np.abs(audio_array))
            if max_val > 0:
                normalized_audio = (audio_array / max_val * 32767).astype(np.int16)
            else:
                normalized_audio = audio_array
            
            # Simple noise gate (remove very quiet parts)
            threshold = np.std(normalized_audio) * 0.1
            noise_gated = np.where(np.abs(normalized_audio) < threshold, 0, normalized_audio)
            
            return noise_gated.tobytes()
            
        except Exception as e:
            logger.error(f"Error enhancing audio: {e}")
            return audio_data  # Return original if enhancement fails 