
import json
import re
from typing import Any
from pypdf import PdfReader

def extract_pdf_text(file_obj) -> str:
	reader = PdfReader(file_obj)
	chunks: list[str] = []
	for page in reader.pages:
		text = page.extract_text() or ""
		if text.strip():
			chunks.append(text)
	return "\n\n".join(chunks).strip()


def extract_summary_candidate(resume_text: str) -> str:
	text = resume_text.replace("\r\n", "\n")
	patterns = [
		r"(?im)^(professional\s+summary|summary|profile|about)\s*$",
		r"(?im)^(professional\s+summary|summary|profile|about)\s*:\s*$",
	]
	section_starts: list[int] = []
	for pat in patterns:
		m = re.search(pat, text)
		if m:
			section_starts.append(m.end())
	if not section_starts:
		snippet = re.sub(r"\s+", " ", text).strip()
		return snippet[:800]

	start = min(section_starts)
	tail = text[start:]
	lines = [ln.rstrip() for ln in tail.split("\n")]

	stop_words = re.compile(
		r"(?i)^(experience|work\s+experience|education|skills|projects|certifications|achievements|publications|contact)\b"
	)

	collected: list[str] = []
	for ln in lines:
		stripped = ln.strip()
		if not stripped:
			if collected:
				collected.append("")
			continue
		if stop_words.match(stripped):
			break
		if len(stripped) < 70 and (stripped.isupper() or stripped.endswith(":")):
			break
		collected.append(stripped)
		if sum(len(x) for x in collected) > 900:
			break

	candidate = "\n".join(collected).strip()
	if candidate:
		return candidate

	snippet = re.sub(r"\s+", " ", text).strip()
	return snippet[:800]


def extract_bullet_candidates(resume_text: str, *, max_bullets: int = 8) -> list[str]:
	text = resume_text.replace("\r\n", "\n")
	lines = [ln.strip() for ln in text.split("\n")]
	bullets: list[str] = []
	for ln in lines:
		if not ln:
			continue
		if ln.startswith(("- ", "•", "* ", "\u2022", "\u25cf")):
			clean = ln.lstrip("-*•\u2022\u25cf ").strip()
			if 20 <= len(clean) <= 240:
				bullets.append(clean)
		if len(bullets) >= max_bullets:
			break
	seen: set[str] = set()
	result: list[str] = []
	for b in bullets:
		key = re.sub(r"\s+", " ", b).strip().lower()
		if key and key not in seen:
			seen.add(key)
			result.append(b)
	return result


def trim_for_prompt(text: str, *, max_chars: int) -> str:
	"""Trim long user-provided text to keep prompts bounded."""
	s = (text or "").strip()
	if len(s) <= max_chars:
		return s
	return s[:max_chars].rstrip() + "\n\n[TRUNCATED]"


def extract_first_json_object(text: str) -> dict[str, Any]:
	s = str(text or "").strip()
	if not s:
		raise ValueError("Model returned empty response")

	fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.IGNORECASE | re.DOTALL)
	if fence:
		candidate = fence.group(1).strip()
		try:
			return json.loads(candidate)
		except json.JSONDecodeError:
			pass
		
	for start in (i for i, ch in enumerate(s) if ch == "{"):
		depth = 0
		in_string = False
		escaped = False
		for end in range(start, len(s)):
			ch = s[end]
			if in_string:
				if escaped:
					escaped = False
				elif ch == "\\":
					escaped = True
				elif ch == '"':
					in_string = False
				continue

			if ch == '"':
				in_string = True
				continue
			if ch == "{":
				depth += 1
				continue
			if ch == "}":
				depth -= 1
				if depth == 0:
					candidate = s[start : end + 1]
					try:
						return json.loads(candidate)
					except json.JSONDecodeError:
						break

	raise ValueError("Model did not return valid JSON")
