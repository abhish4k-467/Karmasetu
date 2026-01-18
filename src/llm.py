
import os
from typing import Any
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class GroqLLM:
	def __init__(self, *, api_key: str, model_id: str, temperature: float, max_tokens: int):
		self._client = Groq(api_key=api_key)
		self._model_id = model_id
		self._temperature = temperature
		self._max_tokens = max_tokens

	@staticmethod
	def _wants_json(prompt: str) -> bool:
		p = (prompt or "").lower()
		return (
			"return only valid json" in p
			or "return only json" in p
			or "exact schema" in p
			or "\"missing_keywords\"" in p
			or "\"interview_questions\"" in p
			or "\"skill_vector\"" in p
			or "\"required_skills\"" in p
			or "\"nice_to_have_skills\"" in p
			or "\"gap_report\"" in p
			or "\"rewritten_bullets\"" in p
		)

	@staticmethod
	def _extract_content(resp) -> str:
		try:
			choices = getattr(resp, "choices", None) or []
			if not choices:
				return ""
			msg = getattr(choices[0], "message", None)
			return (getattr(msg, "content", None) or "").strip()
		except Exception:
			return ""

	def invoke(self, prompt: str) -> str:
		messages = [
			{
				"role": "system",
				"content": "Follow the user's instructions precisely. If asked to return JSON, return ONLY valid JSON with no extra text and no markdown code fences.",
			},
			{"role": "user", "content": prompt},
		]

		wants_json = self._wants_json(prompt)
		create_kwargs: dict[str, Any] = {
			"model": self._model_id,
			"messages": messages,
			"temperature": 0 if wants_json else self._temperature,
			"max_tokens": max(self._max_tokens, 1200) if wants_json else self._max_tokens,
		}
		if wants_json:
			create_kwargs["response_format"] = {"type": "json_object"}

		first_error: Exception | None = None

		try:
			resp = self._client.chat.completions.create(**create_kwargs)
		except Exception as e:
			first_error = e
			if "response_format" in create_kwargs:
				try:
					create_kwargs.pop("response_format", None)
					resp = self._client.chat.completions.create(**create_kwargs)
				except Exception as e2:
					raise RuntimeError(f"Groq API request failed: {e2}") from e2
			else:
				raise RuntimeError(f"Groq API request failed: {e}") from e

		content = self._extract_content(resp)
		if content:
			return content

		create_kwargs["temperature"] = 0
		try:
			resp2 = self._client.chat.completions.create(**create_kwargs)
			content2 = self._extract_content(resp2)
			if content2:
				return content2

			finish_reason = ""
			try:
				finish_reason = getattr(getattr(resp2, "choices", [None])[0], "finish_reason", "") or ""
			except Exception:
				finish_reason = ""

			if finish_reason == "length":
				create_kwargs["max_tokens"] = int(create_kwargs.get("max_tokens", self._max_tokens)) + 800
				resp3 = self._client.chat.completions.create(**create_kwargs)
				content3 = self._extract_content(resp3)
				if content3:
					return content3

			raise RuntimeError(
				"Groq returned an empty response. "
				+ (f"finish_reason={finish_reason}. " if finish_reason else "")
				+ "Try increasing GROQ_MAX_TOKENS, switching GROQ_MODEL_ID, or shortening the resume/job description text."
			)
		except Exception as e3:
			if first_error is not None:
				raise RuntimeError(f"Groq API request failed after retry: {e3} (first error: {first_error})") from e3
			raise RuntimeError(f"Groq API request failed after retry: {e3}") from e3


def get_groq_llm() -> GroqLLM:
	api_key = os.getenv("GROQ_API_KEY")
	model_id = os.getenv("GROQ_MODEL_ID", "openai/gpt-oss-20b")
	missing = [k for k, v in {"GROQ_API_KEY": api_key}.items() if not v]
	if missing:
		raise RuntimeError("Missing environment variables: " + ", ".join(missing))

	temperature = float(os.getenv("GROQ_TEMPERATURE", "0.2"))
	try:
		max_tokens = int(os.getenv("GROQ_MAX_TOKENS", "5000"))
	except ValueError:
		max_tokens = 5000
		
	if max_tokens <= 0:
		raise RuntimeError("GROQ_MAX_TOKENS must be > 0")
	return GroqLLM(api_key=api_key or "", model_id=model_id, temperature=temperature, max_tokens=max_tokens)
