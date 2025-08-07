import logging
from typing import Dict, List, Optional
from textblob import TextBlob
import re
import numpy as np

logger = logging.getLogger(__name__)

class SpeechAnalyzer:
    """Service for analyzing speech patterns, sentiment, and communication quality"""
    
    def __init__(self):
        self.filler_words = [
            'um', 'uh', 'like', 'you know', 'basically', 'actually',
            'literally', 'sort of', 'kind of', 'right', 'so', 'well'
        ]
        
        self.confidence_indicators = [
            'definitely', 'certainly', 'absolutely', 'clearly',
            'obviously', 'without a doubt', 'for sure'
        ]
        
        self.hesitation_indicators = [
            'maybe', 'perhaps', 'possibly', 'I think',
            'I guess', 'I suppose', 'not sure'
        ]
    
    def analyze_speech_patterns(self, transcript: str, audio_metrics: Dict = None) -> Dict:
        """Analyze speech patterns and communication quality"""
        
        try:
            # Basic text analysis
            text_analysis = self._analyze_text_patterns(transcript)
            
            # Sentiment analysis
            sentiment_analysis = self._analyze_sentiment(transcript)
            
            # Communication quality
            communication_quality = self._analyze_communication_quality(transcript)
            
            # Combine with audio metrics if available
            if audio_metrics:
                audio_analysis = self._analyze_audio_patterns(audio_metrics)
            else:
                audio_analysis = {}
            
            analysis_result = {
                "text_analysis": text_analysis,
                "sentiment_analysis": sentiment_analysis,
                "communication_quality": communication_quality,
                "audio_analysis": audio_analysis,
                "overall_score": self._calculate_overall_score(
                    text_analysis, sentiment_analysis, communication_quality, audio_analysis
                )
            }
            
            logger.info(f"Speech analysis completed. Overall score: {analysis_result['overall_score']}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error analyzing speech patterns: {e}")
            return {
                "error": str(e),
                "overall_score": 0
            }
    
    def _analyze_text_patterns(self, transcript: str) -> Dict:
        """Analyze text patterns and structure"""
        
        if not transcript:
            return {"error": "Empty transcript"}
        
        # Basic statistics
        words = transcript.lower().split()
        sentences = re.split(r'[.!?]+', transcript)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Word count and length analysis
        word_count = len(words)
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        avg_sentence_length = np.mean([len(s.split()) for s in sentences]) if sentences else 0
        
        # Filler word analysis
        filler_count = sum(1 for word in words if word in self.filler_words)
        filler_ratio = filler_count / word_count if word_count > 0 else 0
        
        # Confidence indicators
        confidence_count = sum(1 for word in words if word in self.confidence_indicators)
        confidence_ratio = confidence_count / word_count if word_count > 0 else 0
        
        # Hesitation indicators
        hesitation_count = sum(1 for word in words if word in self.hesitation_indicators)
        hesitation_ratio = hesitation_count / word_count if word_count > 0 else 0
        
        # Vocabulary diversity
        unique_words = len(set(words))
        vocabulary_diversity = unique_words / word_count if word_count > 0 else 0
        
        return {
            "word_count": word_count,
            "sentence_count": len(sentences),
            "avg_word_length": round(avg_word_length, 2),
            "avg_sentence_length": round(avg_sentence_length, 2),
            "filler_word_count": filler_count,
            "filler_ratio": round(filler_ratio, 3),
            "confidence_indicators": confidence_count,
            "confidence_ratio": round(confidence_ratio, 3),
            "hesitation_indicators": hesitation_count,
            "hesitation_ratio": round(hesitation_ratio, 3),
            "unique_words": unique_words,
            "vocabulary_diversity": round(vocabulary_diversity, 3)
        }
    
    def _analyze_sentiment(self, transcript: str) -> Dict:
        """Analyze sentiment and emotional tone"""
        
        if not transcript:
            return {"error": "Empty transcript"}
        
        try:
            blob = TextBlob(transcript)
            
            # Polarity (-1 to 1, negative to positive)
            polarity = blob.sentiment.polarity
            
            # Subjectivity (0 to 1, objective to subjective)
            subjectivity = blob.sentiment.subjectivity
            
            # Categorize sentiment
            if polarity > 0.3:
                sentiment_category = "positive"
            elif polarity < -0.3:
                sentiment_category = "negative"
            else:
                sentiment_category = "neutral"
            
            # Analyze emotional indicators
            emotional_indicators = self._analyze_emotional_indicators(transcript)
            
            return {
                "polarity": round(polarity, 3),
                "subjectivity": round(subjectivity, 3),
                "sentiment_category": sentiment_category,
                "emotional_indicators": emotional_indicators,
                "confidence_level": "high" if abs(polarity) > 0.5 else "medium"
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return {
                "error": str(e),
                "polarity": 0,
                "subjectivity": 0,
                "sentiment_category": "neutral"
            }
    
    def _analyze_emotional_indicators(self, transcript: str) -> Dict:
        """Analyze specific emotional indicators in speech"""
        
        emotional_words = {
            "enthusiasm": ["excited", "thrilled", "passionate", "love", "great", "amazing"],
            "confidence": ["confident", "sure", "certain", "definitely", "absolutely"],
            "uncertainty": ["maybe", "perhaps", "unsure", "doubt", "not sure"],
            "frustration": ["frustrated", "annoyed", "difficult", "challenging", "problem"],
            "satisfaction": ["satisfied", "happy", "pleased", "successful", "achieved"]
        }
        
        transcript_lower = transcript.lower()
        indicators = {}
        
        for emotion, words in emotional_words.items():
            count = sum(1 for word in words if word in transcript_lower)
            indicators[emotion] = count
        
        return indicators
    
    def _analyze_communication_quality(self, transcript: str) -> Dict:
        """Analyze communication quality and effectiveness"""
        
        if not transcript:
            return {"error": "Empty transcript"}
        
        words = transcript.lower().split()
        sentences = re.split(r'[.!?]+', transcript)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Clarity analysis
        clarity_score = self._calculate_clarity_score(transcript)
        
        # Structure analysis
        structure_score = self._calculate_structure_score(sentences)
        
        # Engagement analysis
        engagement_score = self._calculate_engagement_score(transcript)
        
        # Professionalism analysis
        professionalism_score = self._calculate_professionalism_score(transcript)
        
        return {
            "clarity_score": round(clarity_score, 2),
            "structure_score": round(structure_score, 2),
            "engagement_score": round(engagement_score, 2),
            "professionalism_score": round(professionalism_score, 2),
            "overall_communication_score": round(
                (clarity_score + structure_score + engagement_score + professionalism_score) / 4, 2
            )
        }
    
    def _calculate_clarity_score(self, transcript: str) -> float:
        """Calculate clarity score based on various factors"""
        
        words = transcript.lower().split()
        if not words:
            return 0
        
        # Factors that improve clarity
        clarity_factors = 0
        total_factors = 0
        
        # Sentence length (optimal: 15-20 words)
        sentences = re.split(r'[.!?]+', transcript)
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        if 10 <= avg_sentence_length <= 25:
            clarity_factors += 1
        total_factors += 1
        
        # Vocabulary diversity (higher is better, but not too high)
        unique_words = len(set(words))
        vocabulary_ratio = unique_words / len(words)
        if 0.3 <= vocabulary_ratio <= 0.7:
            clarity_factors += 1
        total_factors += 1
        
        # Filler word ratio (lower is better)
        filler_count = sum(1 for word in words if word in self.filler_words)
        filler_ratio = filler_count / len(words)
        if filler_ratio < 0.1:
            clarity_factors += 1
        total_factors += 1
        
        # Repetition analysis
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        excessive_repetition = sum(1 for freq in word_freq.values() if freq > 3)
        if excessive_repetition < len(words) * 0.05:
            clarity_factors += 1
        total_factors += 1
        
        return (clarity_factors / total_factors) * 100
    
    def _calculate_structure_score(self, sentences: List[str]) -> float:
        """Calculate structure score based on sentence organization"""
        
        if not sentences:
            return 0
        
        structure_factors = 0
        total_factors = 0
        
        # Sentence variety (mix of short and long sentences)
        sentence_lengths = [len(s.split()) for s in sentences]
        length_variance = np.var(sentence_lengths)
        if length_variance > 10:  # Good variety
            structure_factors += 1
        total_factors += 1
        
        # Logical flow indicators
        flow_indicators = ["first", "second", "then", "next", "finally", "therefore", "however"]
        flow_count = sum(1 for sentence in sentences 
                        for indicator in flow_indicators 
                        if indicator in sentence.lower())
        if flow_count > 0:
            structure_factors += 1
        total_factors += 1
        
        # Question-answer structure
        question_count = sum(1 for s in sentences if '?' in s)
        if question_count > 0:
            structure_factors += 1
        total_factors += 1
        
        return (structure_factors / total_factors) * 100
    
    def _calculate_engagement_score(self, transcript: str) -> float:
        """Calculate engagement score based on interactive elements"""
        
        engagement_factors = 0
        total_factors = 0
        
        # Personal pronouns (indicates engagement)
        personal_pronouns = ["i", "me", "my", "we", "our", "us"]
        pronoun_count = sum(1 for word in transcript.lower().split() if word in personal_pronouns)
        if pronoun_count > 0:
            engagement_factors += 1
        total_factors += 1
        
        # Specific examples and details
        example_indicators = ["for example", "specifically", "in particular", "such as"]
        example_count = sum(1 for indicator in example_indicators if indicator in transcript.lower())
        if example_count > 0:
            engagement_factors += 1
        total_factors += 1
        
        # Enthusiasm indicators
        enthusiasm_words = ["excited", "passionate", "love", "great", "amazing", "fantastic"]
        enthusiasm_count = sum(1 for word in transcript.lower().split() if word in enthusiasm_words)
        if enthusiasm_count > 0:
            engagement_factors += 1
        total_factors += 1
        
        return (engagement_factors / total_factors) * 100
    
    def _calculate_professionalism_score(self, transcript: str) -> float:
        """Calculate professionalism score"""
        
        professionalism_factors = 0
        total_factors = 0
        
        # Formal language usage
        formal_words = ["therefore", "consequently", "furthermore", "additionally", "moreover"]
        formal_count = sum(1 for word in transcript.lower().split() if word in formal_words)
        if formal_count > 0:
            professionalism_factors += 1
        total_factors += 1
        
        # Technical terminology (indicates expertise)
        technical_words = ["implementation", "methodology", "framework", "optimization", "analysis"]
        technical_count = sum(1 for word in transcript.lower().split() if word in technical_words)
        if technical_count > 0:
            professionalism_factors += 1
        total_factors += 1
        
        # Avoidance of informal language
        informal_words = ["gonna", "wanna", "gotta", "kinda", "sorta"]
        informal_count = sum(1 for word in transcript.lower().split() if word in informal_words)
        if informal_count == 0:
            professionalism_factors += 1
        total_factors += 1
        
        return (professionalism_factors / total_factors) * 100
    
    def _analyze_audio_patterns(self, audio_metrics: Dict) -> Dict:
        """Analyze audio patterns for communication quality"""
        
        try:
            # Extract relevant metrics
            duration = audio_metrics.get("duration", 0)
            quality_score = audio_metrics.get("quality_score", 0)
            rms_level = audio_metrics.get("rms_level", 0)
            dynamic_range = audio_metrics.get("dynamic_range", 0)
            
            # Audio quality analysis
            audio_analysis = {
                "duration_appropriate": duration >= 10 and duration <= 300,  # 10s to 5min
                "quality_score": quality_score,
                "volume_appropriate": 1000 <= rms_level <= 5000,
                "dynamic_range_good": dynamic_range > 20,
                "overall_audio_quality": "good" if quality_score > 70 else "needs_improvement"
            }
            
            return audio_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing audio patterns: {e}")
            return {"error": str(e)}
    
    def _calculate_overall_score(
        self,
        text_analysis: Dict,
        sentiment_analysis: Dict,
        communication_quality: Dict,
        audio_analysis: Dict
    ) -> float:
        """Calculate overall speech analysis score"""
        
        try:
            scores = []
            weights = []
            
            # Text analysis score (30%)
            if "error" not in text_analysis:
                text_score = 100 - (text_analysis.get("filler_ratio", 0) * 100)
                scores.append(text_score)
                weights.append(0.3)
            
            # Sentiment score (20%)
            if "error" not in sentiment_analysis:
                sentiment_score = (sentiment_analysis.get("polarity", 0) + 1) * 50
                scores.append(sentiment_score)
                weights.append(0.2)
            
            # Communication quality score (40%)
            if "error" not in communication_quality:
                comm_score = communication_quality.get("overall_communication_score", 0)
                scores.append(comm_score)
                weights.append(0.4)
            
            # Audio quality score (10%)
            if "error" not in audio_analysis:
                audio_score = audio_analysis.get("quality_score", 0)
                scores.append(audio_score)
                weights.append(0.1)
            
            # Calculate weighted average
            if scores and weights:
                total_weight = sum(weights)
                weighted_score = sum(score * weight for score, weight in zip(scores, weights))
                return round(weighted_score / total_weight, 2)
            else:
                return 0
                
        except Exception as e:
            logger.error(f"Error calculating overall score: {e}")
            return 0 