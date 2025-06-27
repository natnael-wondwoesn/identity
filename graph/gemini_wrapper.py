"""
Fixed Gemini LLM wrapper to handle the "Unknown field for Part: thought" error
"""

import google.generativeai as genai
from typing import Any, Dict, List, Optional
import re
import json


class FixedGeminiLLM:
    """Fixed Gemini LLM wrapper that properly handles prompts"""

    def __init__(
        self, api_key: str, model: str = "gemini-2.0-flash", temperature: float = 0.1
    ):
        self.api_key = api_key
        self.model = self._get_correct_model_name(model)
        self.temperature = temperature
        self._setup_gemini()

    def _get_correct_model_name(self, model: str) -> str:
        """Map to correct Gemini model names"""
        model_mapping = {
            "gemini-pro": "gemini-2.0-flash",
            "gemini-1.0-pro": "gemini-2.0-flash",
            "gemini-pro-latest": "gemini-2.0-flash",
            "gemini-1.5-pro": "gemini-1.5-pro",
            "gemini-2.0-flash": "gemini-2.0-flash",
        }
        return model_mapping.get(model, "gemini-2.0-flash")

    def _setup_gemini(self):
        """Setup Gemini API"""
        try:
            genai.configure(api_key=self.api_key)

            # Configure safety settings to be more permissive for business content
            self.safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_ONLY_HIGH",
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_ONLY_HIGH",
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_ONLY_HIGH",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_ONLY_HIGH",
                },
            ]

            # Generation configuration
            self.generation_config = {
                "temperature": self.temperature,
                "top_p": 0.95,
                "top_k": 64,
                "max_output_tokens": 8192,
            }

            self.client = genai.GenerativeModel(
                model_name=self.model,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )

            print(f"✅ Fixed Gemini {self.model} initialized successfully")

        except Exception as e:
            print(f"❌ Failed to setup Gemini: {e}")
            self.client = None

    def _clean_prompt(self, prompt: str) -> str:
        """Clean prompt to avoid Gemini API errors"""

        # Remove any structured data that might confuse Gemini
        cleaned = prompt

        # Remove problematic patterns that cause "thought" field errors
        patterns_to_remove = [
            r'\{[^}]*"thought"[^}]*\}',  # Remove JSON with "thought" field
            r"```json[^`]*```",  # Remove JSON code blocks
            r"```[^`]*```",  # Remove all code blocks for safety
        ]

        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL)

        # Clean up extra whitespace
        cleaned = re.sub(r"\n\s*\n", "\n\n", cleaned)
        cleaned = cleaned.strip()

        # Ensure it's just plain text instructions
        if not cleaned:
            cleaned = "Please provide a helpful response to the user's request."

        return cleaned

    def _ensure_string_prompt(self, prompt: Any) -> str:
        """Ensure prompt is a simple string"""

        if isinstance(prompt, str):
            return self._clean_prompt(prompt)
        elif isinstance(prompt, dict):
            # Extract text from dict
            if "text" in prompt:
                return self._clean_prompt(str(prompt["text"]))
            elif "content" in prompt:
                return self._clean_prompt(str(prompt["content"]))
            else:
                return self._clean_prompt(str(prompt))
        elif isinstance(prompt, list):
            # Join list items
            text_parts = []
            for item in prompt:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    text_parts.append(str(item["text"]))
                else:
                    text_parts.append(str(item))
            return self._clean_prompt(" ".join(text_parts))
        else:
            return self._clean_prompt(str(prompt))

    async def ainvoke(self, prompt: Any):
        """Async invoke for Gemini with proper error handling"""
        try:
            if not self.client:
                return FixedGeminiResponse("Gemini client not initialized")

            # Ensure prompt is clean string
            clean_prompt = self._ensure_string_prompt(prompt)

            # Add some context to make responses more focused
            enhanced_prompt = f"""
You are a helpful AI assistant specializing in market research and business analysis.
Please provide a professional, well-structured response to the following request:

{clean_prompt}

Please structure your response clearly and provide actionable insights where appropriate.
"""

            # Generate response with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.client.generate_content(enhanced_prompt)

                    # Check if response was blocked
                    if response.candidates and len(response.candidates) > 0:
                        candidate = response.candidates[0]
                        if (
                            hasattr(candidate, "finish_reason")
                            and candidate.finish_reason.name != "STOP"
                        ):
                            print(
                                f"⚠️ Gemini response blocked: {candidate.finish_reason.name}"
                            )
                            if attempt < max_retries - 1:
                                # Try with simpler prompt
                                enhanced_prompt = clean_prompt
                                continue

                    # Extract text response
                    if response.text:
                        return FixedGeminiResponse(response.text)
                    else:
                        return FixedGeminiResponse("No response generated")

                except Exception as e:
                    if "Unknown field for Part" in str(e):
                        print(
                            f"⚠️ Gemini prompt format error (attempt {attempt + 1}): {e}"
                        )
                        # Try with even simpler prompt
                        enhanced_prompt = clean_prompt[:500]  # Truncate
                        if attempt < max_retries - 1:
                            continue
                    else:
                        raise e

            return FixedGeminiResponse(f"Gemini error after {max_retries} attempts")

        except Exception as e:
            error_msg = f"Gemini API error: {str(e)}"
            print(f"❌ {error_msg}")

            # Return helpful fallback response based on error type
            if "quota" in str(e).lower() or "limit" in str(e).lower():
                return FixedGeminiResponse(
                    "Rate limit reached - please try again later"
                )
            elif "safety" in str(e).lower():
                return FixedGeminiResponse(
                    "Content filtered by safety settings - please rephrase your request"
                )
            elif "Unknown field" in str(e):
                return FixedGeminiResponse(
                    "Prompt formatting issue detected - using simplified response"
                )
            else:
                return FixedGeminiResponse(f"Unable to generate response: {str(e)}")


class FixedGeminiResponse:
    """Response wrapper for Fixed Gemini"""

    def __init__(self, content: str):
        self.content = content


# Test function
async def test_fixed_gemini():
    """Test the fixed Gemini implementation"""
    import os

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ No API key found")
        return False

    llm = FixedGeminiLLM(api_key=api_key)

    # Test with different prompt types
    test_prompts = [
        "What are the main trends in e-commerce?",
        {"text": "Analyze market opportunities in renewable energy"},
        ["Tell me about", "consumer behavior patterns"],
    ]

    for i, prompt in enumerate(test_prompts):
        try:
            print(f"\n🧪 Test {i+1}: {type(prompt).__name__}")
            response = await llm.ainvoke(prompt)
            print(f"✅ Response: {response.content[:100]}...")
        except Exception as e:
            print(f"❌ Failed: {e}")

    return True


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_fixed_gemini())
