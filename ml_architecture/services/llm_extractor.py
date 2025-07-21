from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import re, json

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

class LLMExtractor:
    def __init__(self, model_name=MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            temperature=0.2,
        )

    def extract(self, text):
        prompt = f"""
Extract the following information from the text below (if available):
- Work experience: position, company, duration
- Education: degree, school, year
- Projects: name, description, role
- Certifications: name, organization, year
- Skills

Return the result as valid JSON.

Text:
{text}
"""
        result = self.pipeline(prompt)[0]['generated_text']
        match = re.search(r'\{.*\}', result, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                return {"raw_output": result, "error": "Could not parse JSON"}
        return {"raw_output": result, "error": "No JSON found"}


# llm_extractor = LLMExtractor() 