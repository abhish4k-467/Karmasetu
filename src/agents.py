
import json
import re
from typing import Any
from .utils import trim_for_prompt, extract_first_json_object, extract_summary_candidate, extract_bullet_candidates

def analyst_skill_vector(llm, resume_text: str) -> list[str]:
	resume_text = trim_for_prompt(resume_text, max_chars=16000)
	prompt = (
		"You are Agent 1: The Analyst (Parser). "
		"Align your output with SDG 8 (Decent Work and Economic Growth): emphasize skills relevant to productive employment, fair labor practices, and professional development. "
		"Extract a structured Skill Vector from the RESUME text. "
		"Include technical skills, tools, languages, frameworks, and core domain skills. "
		"Do not invent skills not supported by the resume. "
		"Return only valid JSON with this exact schema: {\"skill_vector\": [\"s1\", \"s2\", \"s3\"]}. "
		"Rules: 10 to 30 items, deduplicate, use short canonical names (e.g., 'Python', 'SQL', 'Pandas', 'Power BI').\n\n"
		"RESUME:\n" + resume_text
	)
	raw = llm.invoke(prompt)
	data = extract_first_json_object(str(raw))
	skills = data.get("skill_vector")
	if not isinstance(skills, list) or not (10 <= len(skills) <= 30):
		if not isinstance(skills, list): 
			raise ValueError("Analyst output invalid: skill_vector must be a list")
		
	cleaned: list[str] = []
	seen: set[str] = set()
	for s in skills or []:
		if not isinstance(s, str):
			continue
		val = re.sub(r"\s+", " ", s).strip()
		key = val.lower()
		if val and key not in seen:
			seen.add(key)
			cleaned.append(val)
	if not cleaned:
		raise ValueError("Analyst output invalid: no valid skills found")
	return cleaned


def scout_jd_vector(llm, job_description: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
	job_description = trim_for_prompt(job_description, max_chars=12000)
	prompt = (
		"You are Agent 2: The Scout (Market Research). "
		"Align your analysis with SDG 8 (Decent Work and Economic Growth). "
		"Analyze the TARGET JOB DESCRIPTION. "
		"Separate REQUIRED skills from NICE-TO-HAVE skills and assign a weight to each keyword. "
		"Weights: 5=critical, 4=important, 3=useful, 2=minor, 1=optional. "
		"Return only valid JSON with this exact schema: "
		"{\"required_skills\": [{\"skill\": \"...\", \"weight\": 1}], \"nice_to_have_skills\": [{\"skill\": \"...\", \"weight\": 1}]}. "
		"Rules: 6 to 15 required skills; 4 to 12 nice-to-have skills; deduplicate; keep skill names short and canonical.\n\n"
		"JOB DESCRIPTION:\n" + job_description
	)
	raw = llm.invoke(prompt)
	data = extract_first_json_object(str(raw))
	required = data.get("required_skills")
	nice = data.get("nice_to_have_skills")
	if not isinstance(required, list) or not isinstance(nice, list):
		raise ValueError("Scout output invalid: required_skills or nice_to_have_skills not lists")

	def _clean_items(items: list[Any]) -> list[dict[str, Any]]:
		out: list[dict[str, Any]] = []
		seen2: set[str] = set()
		for it in items:
			if not isinstance(it, dict):
				continue
			skill = it.get("skill")
			weight = it.get("weight")
			if not isinstance(skill, str):
				continue
			skill2 = re.sub(r"\s+", " ", skill).strip()
			key = skill2.lower()
			if not skill2 or key in seen2:
				continue
			try:
				w = int(weight)
			except Exception:
				continue
			if w < 1 or w > 5:
				continue
			seen2.add(key)
			out.append({"skill": skill2, "weight": w})
		return out

	required2 = _clean_items(required)
	nice2 = _clean_items(nice)
	# Relaxed check: empty list is possible if JD is weird, but usually bad.
	if not required2 and not nice2: 
		# If both are empty, that's suspicious.
		# But maybe the JD was just "Helper needed". We'll allow it but maybe warn?
		# For now, stick to the logic but be slightly more permissive than raising for just empty req.
		pass 
		
	return required2, nice2


def strategist_gap_report(
	llm,
	*,
	resume_skills: list[str],
	required: list[dict[str, Any]],
	nice: list[dict[str, Any]],
) -> dict[str, Any]:
	prompt = (
		"You are Agent 3: The Strategist (Gap Analysis). "
		"Align the gap analysis with SDG 8 (Decent Work and Economic Growth), focusing on skills that support safe, inclusive, and productive employment. "
		"Compare the RESUME SKILL VECTOR against the JD SKILLS. "
		"Produce a Gap Report listing skills that are missing or underrepresented in the resume. "
		"Return only valid JSON with this exact schema: "
		"{\"matched_skills\": [\"...\"], \"gap_report\": [{\"skill\": \"...\", \"type\": \"required\"|\"nice_to_have\", \"weight\": 1, \"reason\": \"...\"}]}. "
		"Rules: include up to 10 gaps, prioritize required skills with higher weight; do not claim a gap if the skill is present in the resume vector.\n\n"
		"RESUME SKILL VECTOR:\n" + json.dumps(resume_skills, ensure_ascii=False) + "\n\n"
		"JD REQUIRED SKILLS:\n" + json.dumps(required, ensure_ascii=False) + "\n\n"
		"JD NICE-TO-HAVE SKILLS:\n" + json.dumps(nice, ensure_ascii=False)
	)
	raw = llm.invoke(prompt)
	data = extract_first_json_object(str(raw))
	matched = data.get("matched_skills")
	gaps = data.get("gap_report")
	if not isinstance(matched, list) or not isinstance(gaps, list):
		raise ValueError("Strategist output invalid: matched_skills or gap_report not lists")
	return {"matched_skills": matched, "gap_report": gaps}


def editor_rewrite(llm, *, job_description: str, resume_text: str, summary_text: str, bullets: list[str]) -> dict[str, Any]:
	job_description = trim_for_prompt(job_description, max_chars=10000)
	resume_text = trim_for_prompt(resume_text, max_chars=12000)
	summary_text = trim_for_prompt(summary_text, max_chars=1200)
	bullets = bullets[:8]
	prompt = (
		"You are Agent 4: The Editor (Content Generation). "
		"Ensure rewrites align with SDG 8 (Decent Work and Economic Growth): emphasize ethical, safe, and inclusive work practices and skills for sustainable growth. "
		"Rewrite content to improve ATS match to the JOB DESCRIPTION, while being ethical: "
		"do NOT invent new skills, employers, titles, metrics, or projects not supported by the RESUME TEXT. "
		"You may rephrase to highlight existing skills and responsibilities more clearly. "
		"Tasks:\n"
		"1) Rewrite the RESUME SUMMARY into 3-5 sentences (third-person, no exaggeration).\n"
		"2) Rewrite the provided bullet points to better reflect JD keywords without changing meaning.\n"
		"Return only valid JSON with this exact schema: {\"new_summary\": \"...\", \"rewritten_bullets\": [\"...\"]}. "
		"Rules: Keep number of rewritten bullets equal to input bullets; if no bullets are provided, return an empty array for rewritten_bullets.\n\n"
		"JOB DESCRIPTION:\n" + job_description + "\n\n"
		"RESUME SUMMARY (DETECTED):\n" + summary_text + "\n\n"
		"BULLETS TO REWRITE (FROM RESUME):\n" + json.dumps(bullets, ensure_ascii=False) + "\n\n"
		"RESUME TEXT (EVIDENCE):\n" + resume_text
	)
	raw = llm.invoke(prompt)
	data = extract_first_json_object(str(raw))
	new_summary = data.get("new_summary")
	rewritten = data.get("rewritten_bullets")
	if not isinstance(new_summary, str) or not new_summary.strip():
		raise ValueError("Editor output invalid: new_summary missing")
	if not isinstance(rewritten, list):
		raise ValueError("Editor output invalid: rewritten_bullets not a list")
	return {"new_summary": new_summary.strip(), "rewritten_bullets": [str(x).strip() for x in rewritten if str(x).strip()]}
