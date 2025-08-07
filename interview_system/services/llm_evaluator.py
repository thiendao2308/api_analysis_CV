import logging
from typing import Dict, List, Optional
import os
import json

logger = logging.getLogger(__name__)

class LLMEvaluator:
    """Service for evaluating interview responses using LLM"""
    
    def __init__(self):
        # Try to initialize OpenAI client, but provide fallback for testing
        try:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.client = OpenAI(api_key=api_key)
                self.use_openai = True
            else:
                self.client = None
                self.use_openai = False
                logger.warning("OPENAI_API_KEY not set, using mock evaluation")
        except ImportError:
            self.client = None
            self.use_openai = False
            logger.warning("OpenAI not available, using mock evaluation")
    
    def evaluate_response(
        self,
        question: Dict,
        user_response: str,
        job_category: str,
        difficulty: str
    ) -> Dict:
        """Evaluate a single interview response"""
        
        try:
            if self.use_openai and self.client:
                # Use real LLM evaluation
                context = self._prepare_evaluation_context(question, user_response, job_category, difficulty)
                evaluation = self._get_llm_evaluation(context)
                structured_evaluation = self._parse_evaluation_response(evaluation, question)
            else:
                # Use mock evaluation for testing
                structured_evaluation = self._mock_evaluation(question, user_response, job_category, difficulty)
            
            logger.info(f"Evaluated response for question {question.get('id', 'unknown')}")
            return structured_evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating response: {e}")
            return self._fallback_evaluation(question, user_response)
    
    def calculate_session_score(self, evaluations: List[Dict]) -> Dict:
        """Calculate overall session score from multiple evaluations"""
        
        try:
            if not evaluations:
                return {"overall_score": 0, "message": "No evaluations provided"}
            
            # Calculate weighted scores
            total_weighted_score = 0
            total_weight = 0
            category_scores = {
                "technical": [],
                "behavioral": [],
                "situational": [],
                "culture_fit": []
            }
            
            for eval_data in evaluations:
                score = eval_data.get('score', 0)
                question_type = eval_data.get('question_type', 'general')
                weight = eval_data.get('weight', 1.0)
                
                total_weighted_score += score * weight
                total_weight += weight
                
                # Categorize scores
                if question_type in category_scores:
                    category_scores[question_type].append(score)
            
            # Calculate overall score
            overall_score = total_weighted_score / total_weight if total_weight > 0 else 0
            
            # Calculate category averages
            category_averages = {}
            for category, scores in category_scores.items():
                if scores:
                    category_averages[category] = sum(scores) / len(scores)
                else:
                    category_averages[category] = 0
            
            # Generate recommendations
            recommendations = self._generate_recommendations(overall_score, category_averages)
            
            # Generate detailed feedback
            detailed_feedback = self._generate_detailed_feedback(evaluations, overall_score)
            
            session_score = {
                "overall_score": round(overall_score, 2),
                "category_scores": category_averages,
                "recommendations": recommendations,
                "detailed_feedback": detailed_feedback,
                "total_questions": len(evaluations),
                "score_breakdown": {
                    "excellent": len([e for e in evaluations if e.get('score', 0) >= 90]),
                    "good": len([e for e in evaluations if 80 <= e.get('score', 0) < 90]),
                    "fair": len([e for e in evaluations if 70 <= e.get('score', 0) < 80]),
                    "needs_improvement": len([e for e in evaluations if e.get('score', 0) < 70])
                }
            }
            
            logger.info(f"Calculated session score: {overall_score}")
            return session_score
            
        except Exception as e:
            logger.error(f"Error calculating session score: {e}")
            return {"overall_score": 0, "error": str(e)}
    
    def _mock_evaluation(self, question: Dict, user_response: str, job_category: str, difficulty: str) -> Dict:
        """Mock evaluation for testing without OpenAI"""
        
        # Simple scoring based on response length and content
        response_length = len(user_response)
        has_technical_terms = any(term in user_response.lower() for term in ['experience', 'project', 'technology', 'development'])
        has_examples = any(word in user_response.lower() for word in ['example', 'instance', 'case', 'worked on'])
        
        # Calculate mock score
        base_score = 70
        if response_length > 100:
            base_score += 10
        if has_technical_terms:
            base_score += 10
        if has_examples:
            base_score += 10
        
        score = min(100, base_score)
        
        return {
            "score": score,
            "accuracy": score * 0.25,
            "specificity": score * 0.2,
            "communication": score * 0.2,
            "logic": score * 0.2,
            "relevance": score * 0.15,
            "feedback": {
                "accuracy": "Good technical knowledge demonstrated" if has_technical_terms else "Could improve technical depth",
                "specificity": "Good use of examples" if has_examples else "Could provide more specific examples",
                "communication": "Clear and well-structured response",
                "logic": "Logical flow of ideas",
                "relevance": "Directly addresses the question"
            },
            "suggestions": [
                "Provide more concrete examples",
                "Include metrics when possible",
                "Show enthusiasm for the role"
            ],
            "overall_rating": "good" if score >= 80 else "fair",
            "question_type": question.get('type', 'general'),
            "question_id": question.get('id', 0),
            "weight": question.get('scoring_weight', 1.0)
        }
    
    def _prepare_evaluation_context(
        self,
        question: Dict,
        user_response: str,
        job_category: str,
        difficulty: str
    ) -> str:
        """Prepare context for LLM evaluation"""
        
        question_type = question.get('type', 'general')
        expected_points = question.get('expected_points', [])
        
        context = f"""
        EVALUATE THE FOLLOWING INTERVIEW RESPONSE:
        
        Job Category: {job_category}
        Question Type: {question_type}
        Difficulty Level: {difficulty}
        Question: {question.get('question', '')}
        Expected Points: {', '.join(expected_points)}
        
        User Response: {user_response}
        
        EVALUATION CRITERIA:
        1. Accuracy (0-25 points): How well does the response address the question?
        2. Specificity (0-20 points): Does the response provide specific examples and details?
        3. Communication (0-20 points): Is the response clear, well-structured, and articulate?
        4. Logic (0-20 points): Is the reasoning logical and well-organized?
        5. Relevance (0-15 points): Does the response stay relevant to the question?
        
        PROVIDE EVALUATION IN THIS FORMAT:
        {{
            "score": <total_score_0-100>,
            "accuracy": <score_0-25>,
            "specificity": <score_0-20>,
            "communication": <score_0-20>,
            "logic": <score_0-20>,
            "relevance": <score_0-15>,
            "feedback": {{
                "accuracy": "<detailed_feedback>",
                "specificity": "<detailed_feedback>",
                "communication": "<detailed_feedback>",
                "logic": "<detailed_feedback>",
                "relevance": "<detailed_feedback>"
            }},
            "suggestions": ["<suggestion1>", "<suggestion2>", "<suggestion3>"],
            "overall_rating": "<excellent/good/fair/needs_improvement>"
        }}
        """
        
        return context
    
    def _get_llm_evaluation(self, context: str) -> str:
        """Get evaluation from LLM"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert interview evaluator. Provide detailed, fair, and constructive feedback."
                    },
                    {
                        "role": "user",
                        "content": context
                    }
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"LLM evaluation error: {e}")
            raise
    
    def _parse_evaluation_response(self, evaluation_text: str, question: Dict) -> Dict:
        """Parse LLM evaluation response"""
        
        try:
            # Try to parse JSON response
            if evaluation_text.strip().startswith('{'):
                evaluation_data = json.loads(evaluation_text)
            else:
                # Fallback parsing for non-JSON responses
                evaluation_data = self._parse_text_evaluation(evaluation_text)
            
            # Add question metadata
            evaluation_data['question_type'] = question.get('type', 'general')
            evaluation_data['question_id'] = question.get('id', 0)
            evaluation_data['weight'] = question.get('scoring_weight', 1.0)
            
            return evaluation_data
            
        except Exception as e:
            logger.error(f"Error parsing evaluation response: {e}")
            return self._fallback_evaluation(question, "Error parsing evaluation")
    
    def _parse_text_evaluation(self, text: str) -> Dict:
        """Parse text-based evaluation response"""
        
        # Simple parsing for non-JSON responses
        lines = text.split('\n')
        evaluation = {
            "score": 75,  # Default score
            "feedback": {},
            "suggestions": [],
            "overall_rating": "good"
        }
        
        for line in lines:
            line = line.strip().lower()
            if "score" in line and any(char.isdigit() for char in line):
                # Extract score
                score = ''.join(filter(str.isdigit, line))
                if score:
                    evaluation["score"] = min(int(score), 100)
            elif "excellent" in line:
                evaluation["overall_rating"] = "excellent"
            elif "good" in line:
                evaluation["overall_rating"] = "good"
            elif "fair" in line:
                evaluation["overall_rating"] = "fair"
            elif "needs improvement" in line or "poor" in line:
                evaluation["overall_rating"] = "needs_improvement"
        
        return evaluation
    
    def _fallback_evaluation(self, question: Dict, user_response: str) -> Dict:
        """Fallback evaluation when LLM fails"""
        
        return {
            "score": 75,
            "accuracy": 18,
            "specificity": 15,
            "communication": 15,
            "logic": 15,
            "relevance": 12,
            "feedback": {
                "accuracy": "Response addresses the question adequately",
                "specificity": "Could provide more specific examples",
                "communication": "Clear communication demonstrated",
                "logic": "Logical flow of ideas",
                "relevance": "Stays relevant to the question"
            },
            "suggestions": [
                "Provide more concrete examples",
                "Include metrics when possible",
                "Show enthusiasm for the role"
            ],
            "overall_rating": "good",
            "question_type": question.get('type', 'general'),
            "question_id": question.get('id', 0),
            "weight": question.get('scoring_weight', 1.0)
        }
    
    def _generate_recommendations(self, overall_score: float, category_scores: Dict) -> List[str]:
        """Generate recommendations based on scores"""
        
        recommendations = []
        
        if overall_score >= 90:
            recommendations.append("Excellent performance! Continue maintaining high standards.")
        elif overall_score >= 80:
            recommendations.append("Strong performance with room for minor improvements.")
        elif overall_score >= 70:
            recommendations.append("Good foundation, focus on specific areas for improvement.")
        else:
            recommendations.append("Consider additional preparation and practice.")
        
        # Category-specific recommendations
        if category_scores.get('technical', 0) < 75:
            recommendations.append("Focus on improving technical depth and knowledge.")
        
        if category_scores.get('behavioral', 0) < 75:
            recommendations.append("Practice behavioral questions using the STAR method.")
        
        if category_scores.get('communication', 0) < 75:
            recommendations.append("Work on clear and structured communication.")
        
        return recommendations
    
    def _generate_detailed_feedback(self, evaluations: List[Dict], overall_score: float) -> str:
        """Generate detailed session feedback"""
        
        if not evaluations:
            return "No evaluations available for detailed feedback."
        
        strengths = []
        areas_for_improvement = []
        
        # Analyze patterns in evaluations
        for eval_data in evaluations:
            score = eval_data.get('score', 0)
            if score >= 85:
                strengths.append(f"Strong performance in {eval_data.get('question_type', 'general')} questions")
            elif score < 70:
                areas_for_improvement.append(f"Needs improvement in {eval_data.get('question_type', 'general')} questions")
        
        feedback = f"Overall Score: {overall_score}/100\n\n"
        
        if strengths:
            feedback += f"Strengths: {', '.join(strengths)}\n\n"
        
        if areas_for_improvement:
            feedback += f"Areas for Improvement: {', '.join(areas_for_improvement)}\n\n"
        
        if overall_score >= 85:
            feedback += "Overall Assessment: Excellent performance demonstrating strong interview skills."
        elif overall_score >= 75:
            feedback += "Overall Assessment: Good performance with solid interview fundamentals."
        else:
            feedback += "Overall Assessment: Room for improvement in interview preparation and delivery."
        
        return feedback 