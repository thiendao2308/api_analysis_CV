import logging
import openai
import os
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SpellError:
    """Thông tin về lỗi chính tả/định dạng"""
    word: str
    line_number: int
    error_type: str  # "spelling" hoặc "formatting"
    suggestion: str
    context: str
    severity: str  # "low", "medium", "high"

@dataclass
class SpellCheckResult:
    """Kết quả kiểm tra chính tả"""
    total_errors: int
    spelling_errors: int
    formatting_errors: int
    errors: List[SpellError]
    overall_score: float  # 0-100
    suggestions: List[str]
    summary: str

class CVSpellChecker:
    """Kiểm tra chính tả và định dạng cho CV sử dụng LLM"""

    def __init__(self):
        self.client = None
        self._init_openai_client()

        # Từ điển tiếng Việt cơ bản để fallback
        self.vietnamese_words = {
            "nguyễn", "trần", "lê", "phạm", "hoàng", "huỳnh", "phan", "vũ", "võ",
            "đặng", "bùi", "đỗ", "hồ", "ngô", "dương", "lý", "đinh", "tô", "lâm",
            "trịnh", "đoàn", "phùng", "kiều", "cao", "tạ", "hà", "tăng", "lưu",
            "tống", "châu", "từ", "hứa", "hồng", "minh", "thành", "công", "thiện",
            "thị", "văn", "đức", "quang", "huy", "tuấn", "dũng", "hùng", "nam",
            "developer", "engineer", "programmer", "analyst", "manager", "specialist",
            "javascript", "python", "java", "csharp", "react", "angular", "vue",
            "nodejs", "express", "django", "aspnet", "sql", "mongodb", "docker",
            "aws", "azure", "git", "github", "agile", "scrum", "kanban"
        }

    def _init_openai_client(self):
        """Khởi tạo OpenAI client"""
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.client = openai.OpenAI(api_key=api_key)
                logger.info("✅ OpenAI client initialized for spell checking")
            else:
                logger.warning("⚠️ OPENAI_API_KEY not found for spell checking")
        except Exception as e:
            logger.error(f"❌ Error initializing OpenAI client for spell checking: {e}")

    def check_cv_spelling(self, cv_text: str) -> SpellCheckResult:
        """Kiểm tra chính tả và định dạng cho CV"""
        try:
            if not self.client:
                logger.warning("⚠️ OpenAI client not available, using fallback spell checking")
                return self._fallback_spell_check(cv_text)

            # Tạo prompt cho LLM để kiểm tra chính tả
            prompt = self._create_spell_check_prompt(cv_text)

            # Gọi OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "Bạn là một chuyên gia kiểm tra chính tả và định dạng tiếng Việt và tiếng Anh. Hãy kiểm tra CV và tìm ra các lỗi chính tả và định dạng. Trả về kết quả theo format JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=1000
            )

            # Xử lý response từ LLM
            llm_response = response.choices[0].message.content
            logger.info(f"🤖 LLM Spell Check Response: {llm_response}")

            # Parse kết quả từ LLM
            spell_check_result = self._parse_llm_spell_check_response(llm_response, cv_text)

            if spell_check_result.total_errors > 0:
                logger.info(f"✅ LLM found {spell_check_result.total_errors} errors in CV")
            else:
                logger.info("✅ LLM found no errors in CV")

            return spell_check_result

        except Exception as e:
            logger.error(f"❌ Error in LLM spell checking: {e}")
            return self._fallback_spell_check(cv_text)

    def _create_spell_check_prompt(self, cv_text: str) -> str:
        """Tạo prompt cho LLM để kiểm tra chính tả"""
        # Lấy 30 dòng đầu để giảm token usage
        lines = cv_text.split('\n')
        cv_preview = '\n'.join(lines[:30])

        prompt = f"""
Hãy kiểm tra chính tả từ ngữ và định dạng của CV sau đây:

CV TEXT:
{cv_preview}

Yêu cầu kiểm tra:
1. Lỗi chính tả tiếng Việt và tiếng Anh (spelling) - chỉ những từ thực sự sai chính tả
2. Lỗi định dạng (formatting) - khoảng trắng thừa, căn lề sai, viết hoa không nhất quán

LƯU Ý QUAN TRỌNG:
- KHÔNG báo lỗi cho tên riêng, tên công ty, tên công nghệ đúng
- KHÔNG báo lỗi cho "Backend Developer", "Frontend Developer" - đây là tên vị trí đúng
- KHÔNG báo lỗi cho "AI/LLM", "ASP.NET" - đây là tên công nghệ đúng
- Chỉ báo lỗi cho những từ thực sự sai chính tả hoặc định dạng rõ ràng sai
- TRƯỜNG "context" PHẢI VIẾT BẰNG TIẾNG VIỆT

Hãy trả về kết quả theo format JSON:
{{
    "total_errors": số_lỗi_tổng,
    "spelling_errors": số_lỗi_chính_tả_thực_sự,
    "formatting_errors": số_lỗi_định_dạng_thực_sự,
    "errors": [
        {{
            "word": "từ_sai_thực_sự",
            "line_number": số_dòng,
            "error_type": "spelling_hoặc_formatting",
            "suggestion": "gợi_ý_sửa",
            "context": "ngữ_cảnh_bằng_tiếng_việt",
            "severity": "mức_độ_nghiêm_trọng"
        }}
    ],
    "overall_score": điểm_tổng_quan_0_100,
    "suggestions": ["gợi_ý_1", "gợi_ý_2"],
    "summary": "tóm_tắt_kết_quả_bằng_tiếng_việt"
}}

Ví dụ context tiếng Việt:
- "Từ này xuất hiện trong phần mô tả kỹ năng"
- "Tên vị trí công việc cần viết hoa đúng cách"
- "Tên công nghệ cần viết đúng chính tả"
- "Phần mô tả dự án có lỗi định dạng"

Tiêu chí đánh giá:
- Điểm 90-100: Hoàn hảo hoặc chỉ có 1-2 lỗi nhỏ
- Điểm 70-89: Khá tốt, có một số lỗi nhỏ
- Điểm 50-69: Trung bình, có nhiều lỗi
- Điểm 0-49: Kém, có nhiều lỗi nghiêm trọng
"""
        return prompt

    def _parse_llm_spell_check_response(self, llm_response: str, cv_text: str) -> SpellCheckResult:
        """Parse response từ LLM cho spell checking"""
        try:
            import json
            import re

            # Tìm JSON trong response
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)

                # Parse errors
                errors = []
                for error_data in data.get('errors', []):
                    error = SpellError(
                        word=error_data.get('word', ''),
                        line_number=error_data.get('line_number', 0),
                        error_type=error_data.get('error_type', 'spelling'),
                        suggestion=error_data.get('suggestion', ''),
                        context=error_data.get('context', ''),
                        severity=error_data.get('severity', 'medium')
                    )
                    errors.append(error)

                return SpellCheckResult(
                    total_errors=data.get('total_errors', 0),
                    spelling_errors=data.get('spelling_errors', 0),
                    formatting_errors=data.get('formatting_errors', 0),
                    errors=errors,
                    overall_score=data.get('overall_score', 100),
                    suggestions=data.get('suggestions', []),
                    summary=data.get('summary', 'Không có lỗi chính tả')
                )

            # Fallback: parse từ text response
            return self._extract_from_text_response(llm_response, cv_text)

        except Exception as e:
            logger.error(f"❌ Error parsing LLM spell check response: {e}")
            return self._extract_from_text_response(llm_response, cv_text)

    def _extract_from_text_response(self, llm_response: str, cv_text: str) -> SpellCheckResult:
        """Trích xuất thông tin từ text response nếu JSON parse thất bại"""
        try:
            lines = llm_response.split('\n')
            errors = []
            total_errors = 0

            # Tìm thông tin về lỗi trong text response
            for line in lines:
                line_lower = line.lower()

                if 'lỗi' in line_lower or 'error' in line_lower:
                    if ':' in line:
                        error_part = line.split(':')[1].strip()
                        if error_part.isdigit():
                            total_errors = int(error_part)

                if 'điểm' in line_lower or 'score' in line_lower:
                    if ':' in line:
                        score_part = line.split(':')[1].strip()
                        try:
                            score = float(re.findall(r'\d+', score_part)[0])
                        except:
                            score = 100

            # Tạo kết quả fallback với context tiếng Việt
            return SpellCheckResult(
                total_errors=total_errors,
                spelling_errors=total_errors // 2,
                formatting_errors=total_errors // 2,
                errors=errors,
                overall_score=score if 'score' in locals() else 100,
                suggestions=["Kiểm tra lại chính tả và định dạng"],
                summary=f"Tìm thấy {total_errors} lỗi chính tả và định dạng trong CV"
            )

        except Exception as e:
            logger.error(f"❌ Error extracting from text response: {e}")
            return SpellCheckResult(
                total_errors=0,
                spelling_errors=0,
                formatting_errors=0,
                errors=[],
                overall_score=100,
                suggestions=["Không thể kiểm tra chính tả"],
                summary="Không thể kiểm tra chính tả"
            )

    def _fallback_spell_check(self, cv_text: str) -> SpellCheckResult:
        """Fallback spell checking khi LLM không khả dụng"""
        try:
            lines = cv_text.split('\n')
            errors = []
            total_errors = 0

            # Danh sách từ đúng không được báo lỗi
            correct_terms = {
                "backend", "frontend", "fullstack", "developer", "engineer", "programmer",
                "javascript", "python", "java", "csharp", "react", "angular", "vue",
                "nodejs", "express", "django", "aspnet", "sql", "mongodb", "docker",
                "aws", "azure", "git", "github", "agile", "scrum", "kanban",
                "ai", "llm", "machine learning", "deep learning", "neural network",
                "information technology", "software technology", "computer science"
            }

            # Kiểm tra cơ bản: tìm từ có thể sai chính tả
            for i, line in enumerate(lines):
                line_lower = line.lower()
                words = line.split()

                for word in words:
                    # Loại bỏ dấu câu và số
                    clean_word = re.sub(r'[^\w\s]', '', word)

                    if len(clean_word) > 2:
                        # Kiểm tra từ tiếng Việt cơ bản
                        if not any(vn_word in clean_word.lower() for vn_word in self.vietnamese_words):
                            # Kiểm tra xem có phải là tên riêng hoặc từ đúng không
                            if not self._is_proper_noun(clean_word) and clean_word.lower() not in correct_terms:
                                # Kiểm tra thêm: không báo lỗi cho các từ có thể đúng
                                if not self._is_likely_correct(clean_word, line):
                                    # Tạo context tiếng Việt dựa trên ngữ cảnh
                                    context_vn = self._create_vietnamese_context(line, i + 1)
                                    
                                    # Có thể là lỗi chính tả
                                    error = SpellError(
                                        word=word,
                                        line_number=i + 1,
                                        error_type="spelling",
                                        suggestion=f"Kiểm tra lại từ '{word}'",
                                        context=context_vn,
                                        severity="low"
                                    )
                                    errors.append(error)
                                    total_errors += 1

            # Tính điểm dựa trên số lỗi
            if total_errors == 0:
                overall_score = 100
            elif total_errors <= 2:
                overall_score = 95
            elif total_errors <= 5:
                overall_score = 85
            elif total_errors <= 10:
                overall_score = 75
            else:
                overall_score = 65

            return SpellCheckResult(
                total_errors=total_errors,
                spelling_errors=total_errors,
                formatting_errors=0,
                errors=errors,
                overall_score=overall_score,
                suggestions=["Sử dụng LLM để kiểm tra chính tả chính xác hơn"],
                summary=f"Fallback check: Tìm thấy {total_errors} lỗi chính tả"
            )

        except Exception as e:
            logger.error(f"❌ Error in fallback spell checking: {e}")
            return SpellCheckResult(
                total_errors=0,
                spelling_errors=0,
                formatting_errors=0,
                errors=[],
                overall_score=100,
                suggestions=["Không thể kiểm tra chính tả"],
                summary="Không thể kiểm tra chính tả"
            )

    def _is_proper_noun(self, word: str) -> bool:
        """Kiểm tra xem từ có phải là danh từ riêng không"""
        # Danh từ riêng thường viết hoa
        if word[0].isupper() and len(word) > 2:
            return True

        # Tên công ty, công nghệ phổ biến
        tech_terms = {
            "microsoft", "google", "facebook", "amazon", "apple", "netflix",
            "react", "angular", "vue", "node", "python", "java", "javascript",
            "html", "css", "sql", "mongodb", "docker", "kubernetes", "aws",
            "azure", "github", "gitlab", "bitbucket", "jira", "confluence"
        }

        if word.lower() in tech_terms:
            return True

        return False

    def _is_likely_correct(self, word: str, context: str) -> bool:
        """Kiểm tra xem từ có khả năng đúng không dựa trên ngữ cảnh"""
        word_lower = word.lower()
        context_lower = context.lower()
        
        # Các pattern cho từ có thể đúng
        patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Proper nouns like "Backend Developer"
            r'\b[A-Z]{2,}\b',  # Acronyms like "AI", "LLM", "API"
            r'\b[A-Za-z]+\.NET\b',  # .NET technologies
            r'\b[A-Za-z]+\.js\b',  # JavaScript frameworks
            r'\b[A-Za-z]+\.py\b',  # Python frameworks
        ]
        
        for pattern in patterns:
            if re.match(pattern, word):
                return True
        
        # Kiểm tra ngữ cảnh
        if any(tech in context_lower for tech in ['technology', 'framework', 'library', 'tool']):
            return True
            
        if any(company in context_lower for company in ['company', 'corporation', 'inc', 'ltd']):
            return True
            
        return False

    def _create_vietnamese_context(self, line: str, line_number: int) -> str:
        """Tạo context tiếng Việt cho lỗi chính tả"""
        line_lower = line.lower()
        
        # Xác định loại nội dung dựa trên từ khóa
        if any(word in line_lower for word in ['kỹ năng', 'skill', 'technology', 'framework']):
            return f"Từ này xuất hiện trong phần mô tả kỹ năng (dòng {line_number})"
        elif any(word in line_lower for word in ['vị trí', 'position', 'job', 'title']):
            return f"Từ này xuất hiện trong phần mô tả vị trí công việc (dòng {line_number})"
        elif any(word in line_lower for word in ['dự án', 'project', 'description']):
            return f"Từ này xuất hiện trong phần mô tả dự án (dòng {line_number})"
        elif any(word in line_lower for word in ['công ty', 'company', 'organization']):
            return f"Từ này xuất hiện trong phần thông tin công ty (dòng {line_number})"
        elif any(word in line_lower for word in ['học vấn', 'education', 'degree', 'school']):
            return f"Từ này xuất hiện trong phần thông tin học vấn (dòng {line_number})"
        else:
            return f"Từ này xuất hiện trong nội dung CV (dòng {line_number})"
