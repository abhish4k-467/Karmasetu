
import streamlit as st
from typing import Any
from src.utils import extract_pdf_text, extract_summary_candidate, extract_bullet_candidates
from src.llm import get_groq_llm
from src.agents import analyst_skill_vector, scout_jd_vector, strategist_gap_report, editor_rewrite

st.set_page_config(page_title="KarmaSetu", layout="wide")
st.title("KARMA SETU : Bridge to career")
st.subheader("An Intelligent Career Strategist🏢")


def _render_skill_vector(skills: list[str]) -> None:
	if not skills:
		st.info("No skills detected.")
		return
	st.caption(f"{len(skills)} skills")
	st.markdown("\n".join(f"- {s}" for s in skills))


def _render_weighted_skills(items: list[dict[str, Any]]) -> None:
	rows: list[dict[str, Any]] = []
	for it in items or []:
		if not isinstance(it, dict):
			continue
		skill = it.get("skill")
		weight = it.get("weight")
		if isinstance(skill, str) and skill.strip():
			rows.append({"Skill": skill.strip(), "Weight": int(weight) if isinstance(weight, int) else weight})
	if not rows:
		st.info("No items.")
		return
	rows = sorted(rows, key=lambda r: (-(r.get("Weight") or 0), str(r.get("Skill") or "")))
	st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_gap_report(gaps: list[dict[str, Any]]) -> None:
	rows: list[dict[str, Any]] = []
	for g in gaps or []:
		if not isinstance(g, dict):
			continue
		skill = g.get("skill")
		gap_type = g.get("type")
		weight = g.get("weight")
		reason = g.get("reason")
		if not isinstance(skill, str) or not skill.strip():
			continue
		rows.append(
			{
				"Skill": skill.strip(),
				"Type": str(gap_type or "").strip(),
				"Weight": int(weight) if isinstance(weight, int) else weight,
				"Reason": str(reason or "").strip(),
			}
		)
	if not rows:
		st.success("No gaps detected.")
		return
	rows = sorted(rows, key=lambda r: (r.get("Type") != "required", -(r.get("Weight") or 0), r.get("Skill") or ""))
	st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_bullets_side_by_side(original: list[str], rewritten: list[str]) -> None:
	original = original or []
	rewritten = rewritten or []
	if not original:
		st.info("No bullet points detected in resume text.")
		return
	rows: list[dict[str, Any]] = []
	for i, orig in enumerate(original):
		rows.append({"Original": orig, "Rewritten": rewritten[i] if i < len(rewritten) else ""})
	st.dataframe(rows, use_container_width=True, hide_index=True)

col1, col2 = st.columns(2)
with col1:
	resume_pdf = st.file_uploader("Upload PDF Resume", type=["pdf"])
with col2:
	job_description = st.text_area("Paste Job Description", height=260)

run = st.button("Run Agents", type="primary", disabled=not resume_pdf or not job_description.strip())

if run:
	try:
		resume_text = extract_pdf_text(resume_pdf)
		if not resume_text.strip():
			st.error("Could not extract text from the PDF.")
			st.stop()

		llm = get_groq_llm()

		with st.spinner("Agent 1 (Analyst): parsing resume into a Skill Vector..."):
			resume_skills = analyst_skill_vector(llm, resume_text=resume_text)

		with st.spinner("Agent 2 (Scout): analyzing job description..."):
			required_skills, nice_to_have_skills = scout_jd_vector(llm, job_description=job_description)

		with st.spinner("Agent 3 (Strategist): generating gap report..."):
			gap = strategist_gap_report(llm, resume_skills=resume_skills, required=required_skills, nice=nice_to_have_skills)

		summary_candidate = extract_summary_candidate(resume_text)
		bullet_candidates = extract_bullet_candidates(resume_text, max_bullets=8)

		with st.spinner("Agent 4 (Editor): rewriting summary and bullet points..."):
			edited = editor_rewrite(
				llm,
				job_description=job_description,
				resume_text=resume_text,
				summary_text=summary_candidate,
				bullets=bullet_candidates,
			)

		st.subheader("Resume Skill Vector")
		_render_skill_vector(resume_skills)

		left, right = st.columns(2)
		with left:
			st.subheader("JD Required Skills")
			_render_weighted_skills(required_skills)
		with right:
			st.subheader("JD Nice-to-have Skills")
			_render_weighted_skills(nice_to_have_skills)

		st.subheader("Gap Report")
		_render_gap_report(gap.get("gap_report", []))
		with st.expander("Matched Skills"):
			_render_skill_vector(gap.get("matched_skills", []))

		left2, right2 = st.columns(2)
		with left2:
			st.subheader("Original Summary (Detected)")
			st.text_area("original_summary", value=summary_candidate, height=220, label_visibility="collapsed")
		with right2:
			st.subheader("Rewritten Summary")
			st.text_area("rewritten_summary", value=edited.get("new_summary", ""), height=220, label_visibility="collapsed")

		st.subheader("Bullet Points")
		_render_bullets_side_by_side(bullet_candidates, edited.get("rewritten_bullets", []))
		
		with st.expander("Extracted Resume Text"):
			st.text_area("extracted_resume_text", value=resume_text, height=260, label_visibility="collapsed")
	except Exception as e:
		st.error(str(e))
