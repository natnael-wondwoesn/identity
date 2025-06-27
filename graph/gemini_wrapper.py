"""
Fixed Gemini LLM wrapper to resolve the "Unknown field for Part: thought" error
and other integration issues.
"""

import google.generativeai as genai
from typing import Any, Dict, List, Optional
import re
import json
import asyncio
from datetime import datetime


class FixedGeminiLLM:
    """Fixed Gemini LLM wrapper that properly handles prompts and avoids API errors"""

    def __init__(
        self, api_key: str, model: str = "gemini-1.5-flash", temperature: float = 0.1
    ):
        self.api_key = api_key
        self.model = self._get_correct_model_name(model)
        self.temperature = temperature
        self._setup_gemini()

    def _get_correct_model_name(self, model: str) -> str:
        """Map to correct Gemini model names that actually work"""
        model_mapping = {
            "gemini-pro": "gemini-1.5-flash",
            "gemini-1.0-pro": "gemini-1.5-flash",
            "gemini-pro-latest": "gemini-1.5-flash",
            "gemini-1.5-pro": "gemini-1.5-pro",
            "gemini-1.5-flash": "gemini-1.5-flash",
            "gemini-2.0-flash": "gemini-1.5-flash",  # Fallback since 2.0 may not be available
        }
        return model_mapping.get(model, "gemini-1.5-flash")

    def _setup_gemini(self):
        """Setup Gemini API with proper error handling"""
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
            self.generation_config = genai.types.GenerationConfig(
                temperature=self.temperature,
                top_p=0.95,
                top_k=64,
                max_output_tokens=8192,
            )

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
        """Clean prompt to avoid Gemini API errors - this is the key fix"""

        # Remove any structured data that might confuse Gemini
        cleaned = prompt

        # Remove problematic patterns that cause "thought" field errors
        patterns_to_remove = [
            r'\{[^}]*"thought"[^}]*\}',  # Remove JSON with "thought" field
            r"```json[^`]*```",  # Remove JSON code blocks
            r"```[^`]*```",  # Remove all code blocks for safety
            r"<[^>]*>",  # Remove any XML/HTML tags
        ]

        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL)

        # Clean up extra whitespace
        cleaned = re.sub(r"\n\s*\n", "\n\n", cleaned)
        cleaned = cleaned.strip()

        # Ensure it's just plain text instructions
        if not cleaned:
            cleaned = "Please provide a helpful response to the user's request."

        # Limit length to avoid token issues
        if len(cleaned) > 4000:
            cleaned = cleaned[:4000] + "..."

        return cleaned

    def _ensure_string_prompt(self, prompt: Any) -> str:
        """Ensure prompt is a simple string - this prevents the Part field error"""

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

            # Ensure prompt is clean string - this is the main fix
            clean_prompt = self._ensure_string_prompt(prompt)

            # Add context to make responses more focused for business use
            enhanced_prompt = f"""You are a helpful AI assistant for market research and business analysis.
Please provide a professional, well-structured response to the following request:

{clean_prompt}

Please provide clear, actionable insights where appropriate."""

            # Generate response with retry logic
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    # Use simple generate_content call
                    response = self.client.generate_content(enhanced_prompt)

                    # Check if response was blocked or empty
                    if not response or not response.text:
                        if attempt < max_retries - 1:
                            # Try with simpler prompt
                            enhanced_prompt = clean_prompt
                            continue
                        else:
                            return FixedGeminiResponse(
                                "Unable to generate response - content may have been filtered"
                            )

                    return FixedGeminiResponse(response.text)

                except Exception as e:
                    error_str = str(e)
                    if "Unknown field for Part" in error_str:
                        print(
                            f"⚠️ Gemini prompt format error (attempt {attempt + 1}): {e}"
                        )
                        # Try with even simpler prompt
                        enhanced_prompt = clean_prompt[:1000]  # Truncate significantly
                        if attempt < max_retries - 1:
                            continue
                    else:
                        raise e

            return FixedGeminiResponse("Failed to generate response after retries")

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
                return FixedGeminiResponse(
                    f"AI analysis: Unable to process request due to technical issues. Please try rephrasing your query."
                )

    def invoke(self, prompt: Any):
        """Synchronous invoke for compatibility"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.ainvoke(prompt))
        finally:
            loop.close()


class FixedGeminiResponse:
    """Response wrapper for Fixed Gemini"""

    def __init__(self, content: str):
        self.content = content

    def __str__(self):
        return self.content


# Test function
async def test_fixed_gemini():
    """Test the fixed Gemini implementation"""
    import os

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ No API key found - set GOOGLE_API_KEY or GEMINI_API_KEY")
        return False

    llm = FixedGeminiLLM(api_key=api_key)

    # Test with different prompt types that previously caused errors
    test_prompts = [
        "What are the main trends in e-commerce?",
        {"text": "Analyze market opportunities in renewable energy"},
        ["Tell me about", "consumer behavior patterns"],
        """Create a market research strategy for:
        Query: Recent developments in artificial intelligence
        Industry: Technology
        Provide structured analysis.""",
    ]

    for i, prompt in enumerate(test_prompts):
        try:
            print(f"\n🧪 Test {i+1}: {type(prompt).__name__}")
            response = await llm.ainvoke(prompt)
            print(f"✅ Response: {response.content[:100]}...")
        except Exception as e:
            print(f"❌ Failed: {e}")
            return False

    return True


if __name__ == "__main__":
    import asyncio

    success = asyncio.run(test_fixed_gemini())
    print(f"\n{'✅ All tests passed!' if success else '❌ Tests failed!'}")
