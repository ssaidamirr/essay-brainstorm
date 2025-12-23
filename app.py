import os
import re
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import streamlit as st

# Optional Gemini support (app still works without it)
try:
    import google.generativeai as genai  # type: ignore
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False


APP_TITLE = "Common App Personal Statement Coach"
APP_SUBTITLE = "A step-by-step guide that helps you brainstorm, choose a story, and build an outline. It does not write your essay."

NO_EM_DASH = True  # keep this true if you want to block em dashes in user-facing outputs


# ----------------------------
# Helpers
# ----------------------------
def strip_em_dashes(text: str) -> str:
    if not text:
        return text
    # Replace em dash and common variants with " - "
    text = text.replace("—", " - ").replace("–", " - ").replace("−", "-")
    # Collapse weird spacing
    text = re.sub(r"\s*-\s*", " - ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def safe_text(text: str) -> str:
    text = text or ""
    if NO_EM_DASH:
        text = strip_em_dashes(text)
    return text


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w']+\b", text or ""))


def clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))


def now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def ensure_state():
    defaults = {
        "step": "Start",
        "profile": {
            "name": "",
            "grade": "",
            "intended_major": "",
            "college_level": "Undergrad",
            "remember_words": "",
            "avoid_topics": "",
            "boundaries": "",
            "word_limit": 650,
        },
        "mined_ideas": {
            "A": [],  # moments that changed you
            "B": [],  # contradictions
            "C": [],  # patterns
            "D": [],  # objects/places
        },
        "idea_details": {},   # idea_id -> details
        "top_candidates": [], # list of (idea_id, score_dict, total)
        "selected_idea_id": None,
        "core_argument": "",
        "scene_bank": [],     # list of scenes
        "outline_template": "Classic Narrative Arc",
        "outline": "",
        "draft_text": "",
        "coach_chat": [],     # list of {"role": "user"/"assistant", "content": "..."}
        "gemini": {
            "use_gemini": False,
            "api_key": "",
            "model_name": "",
            "models_cache": [],
            "last_model_refresh": "",
        },
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def idea_id(bucket: str, idx: int) -> str:
    return f"{bucket}-{idx}"


def bucket_label(bucket: str) -> str:
    return {
        "A": "Moments that changed you",
        "B": "Contradictions and surprises",
        "C": "Patterns in how you act",
        "D": "Objects and places (scene fuel)",
    }[bucket]


def format_candidate_title(idea_text: str, idea_id_: str) -> str:
    short = idea_text.strip()
    if len(short) > 80:
        short = short[:77].rstrip() + "..."
    return f"{short}  ({idea_id_})"


def build_packet() -> str:
    p = st.session_state["profile"]
    mined = st.session_state["mined_ideas"]
    selected_id = st.session_state["selected_idea_id"]
    details = st.session_state["idea_details"].get(selected_id, {}) if selected_id else {}
    candidates = st.session_state["top_candidates"]
    scenes = st.session_state["scene_bank"]
    outline = st.session_state["outline"]
    core = st.session_state["core_argument"]

    def section(title: str) -> str:
        return f"\n{title}\n" + ("=" * len(title)) + "\n"

    out = []
    out.append(f"{APP_TITLE}\nGenerated: {now_stamp()}\n")

    out.append(section("Student Profile"))
    out.append(f"Name: {safe_text(p.get('name',''))}\n")
    out.append(f"Level: {safe_text(p.get('college_level','Undergrad'))}\n")
    out.append(f"Grade: {safe_text(p.get('grade',''))}\n")
    out.append(f"Intended major (optional): {safe_text(p.get('intended_major',''))}\n")
    out.append(f"Target word limit: {p.get('word_limit',650)}\n")
    out.append(f"3 words you want remembered: {safe_text(p.get('remember_words',''))}\n")
    out.append(f"What you want to avoid: {safe_text(p.get('avoid_topics',''))}\n")
    out.append(f"Boundaries: {safe_text(p.get('boundaries',''))}\n")

    out.append(section("Mined Ideas (Raw Material)"))
    for b in ["A", "B", "C", "D"]:
        out.append(f"{bucket_label(b)}:\n")
        for i, idea in enumerate(mined[b], start=1):
            out.append(f"  {b}{i}. {safe_text(idea)}\n")
        out.append("\n")

    out.append(section("Top Candidates (Ranked)"))
    if not candidates:
        out.append("No ranked candidates yet.\n")
    else:
        for rank, (iid, score_dict, total) in enumerate(candidates, start=1):
            txt = st.session_state["idea_details"].get(iid, {}).get("idea_text", "")
            out.append(f"{rank}. {safe_text(txt)} ({iid})\n")
            out.append(f"   Total: {total}/25 | Specificity {score_dict['specificity']}, Depth {score_dict['depth']}, Agency {score_dict['agency']}, Originality {score_dict['originality']}, Values {score_dict['values']}\n")

    out.append(section("Chosen Story"))
    if not selected_id:
        out.append("Not selected yet.\n")
    else:
        out.append(f"Selected idea: {safe_text(details.get('idea_text',''))} ({selected_id})\n\n")
        out.append("Scene Test:\n")
        out.append(f"  Where/who/what: {safe_text(details.get('scene_where',''))}\n")
        out.append(f"  Tension/stakes: {safe_text(details.get('scene_tension',''))}\n")
        out.append(f"  Decision/action: {safe_text(details.get('scene_decision',''))}\n")
        out.append(f"  Change afterward: {safe_text(details.get('scene_change',''))}\n\n")

        out.append("Meaning:\n")
        out.append(f"  Belief before: {safe_text(details.get('belief_before',''))}\n")
        out.append(f"  Belief now: {safe_text(details.get('belief_now',''))}\n")
        out.append(f"  Skill/habit built: {safe_text(details.get('skill_built',''))}\n")
        out.append(f"  How you act differently today: {safe_text(details.get('behavior_now',''))}\n")
        out.append(f"  Values: {safe_text(', '.join(details.get('values_chosen', [])))}\n")

    out.append(section("Core Argument (One Sentence)"))
    out.append(safe_text(core) + "\n")

    out.append(section("Scene Bank (4-6 Scenes)"))
    if not scenes:
        out.append("No scenes added yet.\n")
    else:
        for i, s in enumerate(scenes, start=1):
            out.append(f"Scene {i}:\n")
            out.append(f"  What happened (2-3 lines): {safe_text(s.get('what',''))}\n")
            out.append(f"  What it reveals: {safe_text(s.get('reveals',''))}\n")
            out.append(f"  Connection to core argument: {safe_text(s.get('connects',''))}\n\n")

    out.append(section("Outline"))
    out.append(safe_text(outline) + "\n")

    out.append(section("Drafting Checklist"))
    out.append(
        safe_text(
            "Write in concrete nouns and simple verbs. Show scenes before you explain them. "
            "Cut repeated points. Avoid generic openings. End with forward motion: what you do now because of what changed."
        )
        + "\n"
    )

    return "".join(out)


def build_outline(template_name: str, core: str, scenes: List[Dict[str, str]]) -> str:
    core = safe_text(core).strip()
    # Keep outlines as structured text, not prose
    s_lines = []
    for i, s in enumerate(scenes, start=1):
        s_lines.append(f"Scene {i}: {safe_text(s.get('what','')).strip()}")
    scenes_block = "\n".join([ln for ln in s_lines if ln.strip()])

    if template_name == "Classic Narrative Arc":
        return safe_text(
            f"Hook (start inside a moment)\n"
            f"- Choose one high-tension scene from your bank\n\n"
            f"Context (2-4 lines)\n"
            f"- What the reader must know to understand the hook\n\n"
            f"Challenge\n"
            f"- Name the conflict and why it mattered\n\n"
            f"Turning point\n"
            f"- The decision you made and why\n\n"
            f"Growth shown through actions\n"
            f"- 2-3 examples of what you did differently afterward\n\n"
            f"Now and forward direction\n"
            f"- Where this shows up today and what it suggests about your next environment\n\n"
            f"Core argument:\n- {core}\n\n"
            f"Scene bank (for reference):\n{scenes_block}\n"
        )

    if template_name == "Montage":
        return safe_text(
            f"Theme line (1 sentence)\n"
            f"- {core}\n\n"
            f"Mini-scene 1\n"
            f"- What happened\n- What it reveals\n\n"
            f"Mini-scene 2\n"
            f"- What happened\n- What it reveals\n\n"
            f"Mini-scene 3\n"
            f"- What happened\n- What it reveals\n\n"
            f"Tie-back\n"
            f"- What pattern connects these moments\n\n"
            f"Now\n"
            f"- How you live this today\n\n"
            f"Scene bank (for reference):\n{scenes_block}\n"
        )

    # Problem-solver arc
    return safe_text(
        f"Problem you noticed\n"
        f"- What was broken or missing\n\n"
        f"What you tried\n"
        f"- First attempt (and why)\n\n"
        f"What failed\n"
        f"- What did not work and what you learned\n\n"
        f"What you changed\n"
        f"- The improved approach and your role\n\n"
        f"Impact\n"
        f"- What changed for others or for you\n\n"
        f"Now\n"
        f"- How this pattern shows up today\n\n"
        f"Core argument:\n- {core}\n\n"
        f"Scene bank (for reference):\n{scenes_block}\n"
    )


# ----------------------------
# Gemini coach (optional)
# ----------------------------
COACH_SYSTEM = safe_text(
    "You are an admissions writing coach for the Common App personal statement. "
    "You must not write the student's essay or produce full paragraphs in their voice. "
    "You can do: ask sharp questions, point out vagueness, suggest what to add or cut, propose outline moves, "
    "and provide 2-3 example sentence starters (not a full paragraph). "
    "If the student asks you to write the essay, refuse briefly and continue coaching. "
    "Keep feedback concrete: scenes, stakes, decisions, behavior change. "
    "Avoid em dashes."
)

COACH_STYLE = safe_text(
    "Coaching style: direct, structured, specific. Avoid clichés. Avoid moralizing. "
    "Prioritize: agency, specificity, depth. "
    "Always end your response with 2-4 questions that help the student move forward."
)


def gemini_list_models(api_key: str) -> List[str]:
    if not GEMINI_AVAILABLE or not api_key:
        return []
    try:
        genai.configure(api_key=api_key)
        models = genai.list_models()
        names = []
        for m in models:
            # Only allow generateContent-capable models
            if hasattr(m, "supported_generation_methods") and "generateContent" in m.supported_generation_methods:
                names.append(m.name)
        # Prefer flash/faster models first
        names_sorted = sorted(
            names,
            key=lambda x: (
                0 if "flash" in x.lower() else 1,
                0 if "2.0" in x else 1,
                x.lower(),
            ),
        )
        return names_sorted
    except Exception:
        return []


def gemini_coach(api_key: str, model_name: str, messages: List[Dict[str, str]]) -> str:
    if not GEMINI_AVAILABLE:
        return "Gemini is not installed in this environment. Add google-generativeai to requirements, or use the built-in coach."

    if not api_key:
        return "No API key found. Add it in the sidebar or set it as an environment variable named GOOGLE_API_KEY."

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name=model_name)

        # Convert chat history to a single prompt while keeping roles
        convo = []
        convo.append(f"System:\n{COACH_SYSTEM}\n{COACH_STYLE}\n")
        for m in messages[-12:]:
            role = "Student" if m["role"] == "user" else "Coach"
            convo.append(f"{role}: {m['content']}")
        prompt = "\n\n".join(convo).strip()

        resp = model.generate_content(prompt)
        text = getattr(resp, "text", "") or ""
        return safe_text(text).strip() or "I could not generate a response. Try again."
    except Exception as e:
        return safe_text(f"Gemini error: {e}")


def local_coach(messages: List[Dict[str, str]]) -> str:
    # Simple coaching logic when no LLM: ask for scenes, stakes, decision, change, plus clarity/cuts
    last = ""
    for m in reversed(messages):
        if m["role"] == "user":
            last = m["content"]
            break
    last = safe_text(last)

    # If they pasted a draft, do light critique without rewriting
    if word_count(last) > 120:
        return safe_text(
            "I will not rewrite this for you, but I can help you improve it.\n\n"
            "What I notice:\n"
            "1) Identify the main scene. Right now it reads like summary in places.\n"
            "2) Replace vague lines with concrete actions: what did you do, say, build, decide.\n"
            "3) Reduce lesson statements. Show behavior change instead.\n\n"
            "Try this next:\n"
            "- Pick one paragraph and label it: Scene, Context, Reflection, or Now.\n"
            "- If you have two Reflection paragraphs in a row, cut one.\n\n"
            "Questions:\n"
            "1) What is the single moment where the essay turns?\n"
            "2) What decision did you make that changed what happened next week?\n"
            "3) What do you do today that your past self would not do?\n"
        )

    # Otherwise: brainstorming guidance
    return safe_text(
        "Good. Let us sharpen this into scenes.\n\n"
        "Answer in 2-4 lines each.\n"
        "1) Where are you, and who is present in the moment you want to write about?\n"
        "2) What is the tension or stake? What could go wrong?\n"
        "3) What did you choose to do, not feel?\n"
        "4) What changed afterward in your behavior, not your beliefs?\n"
    )


# ----------------------------
# UI sections
# ----------------------------
def sidebar():
    st.sidebar.title("Settings")

    st.sidebar.caption("Optional: Gemini coach. App works without it.")
    use_g = st.sidebar.toggle("Use Gemini for the coach chat", value=st.session_state["gemini"]["use_gemini"])
    st.session_state["gemini"]["use_gemini"] = use_g

    api_key_env = os.getenv("GOOGLE_API_KEY", "")
    api_key = st.sidebar.text_input(
        "Google API key (or set GOOGLE_API_KEY env var)",
        value=st.session_state["gemini"]["api_key"] or api_key_env,
        type="password",
        disabled=not GEMINI_AVAILABLE,
        help="Used only for the coaching chat. This app still does not write essays.",
    )
    st.session_state["gemini"]["api_key"] = api_key

    if GEMINI_AVAILABLE:
        col1, col2 = st.sidebar.columns([1, 1])
        with col1:
            if st.button("Refresh models"):
                models = gemini_list_models(api_key)
                st.session_state["gemini"]["models_cache"] = models
                st.session_state["gemini"]["last_model_refresh"] = now_stamp()
        with col2:
            st.caption(st.session_state["gemini"]["last_model_refresh"] or "")

        models = st.session_state["gemini"]["models_cache"]
        if use_g and api_key and not models:
            models = gemini_list_models(api_key)
            st.session_state["gemini"]["models_cache"] = models
            st.session_state["gemini"]["last_model_refresh"] = now_stamp()

        if models:
            model_name = st.sidebar.selectbox(
                "Model",
                options=models,
                index=0,
            )
            st.session_state["gemini"]["model_name"] = model_name
        else:
            st.session_state["gemini"]["model_name"] = ""
            if use_g and api_key:
                st.sidebar.warning("No models found. Click Refresh models.")
    else:
        st.sidebar.info("Gemini library not installed. Add google-generativeai to requirements to enable it.")

    st.sidebar.divider()

    st.sidebar.caption("Export")
    packet_text = build_packet()
    st.sidebar.download_button(
        "Download Personal Statement Packet (.txt)",
        data=packet_text.encode("utf-8"),
        file_name="personal_statement_packet.txt",
        mime="text/plain",
    )


def section_start():
    st.title(APP_TITLE)
    st.write(APP_SUBTITLE)

    with st.expander("How this app works (and what it will not do)", expanded=False):
        st.write(
            safe_text(
                "You will move through: story mining, scene testing, meaning, ranking, choosing a core argument, building a scene bank, and producing an outline. "
                "The coach chat can critique and ask questions, but it will not write your essay for you."
            )
        )

    st.session_state["profile"]["word_limit"] = clamp(int(st.session_state["profile"]["word_limit"]), 250, 1000)

    st.subheader("Step 1: Quick profile")
    p = st.session_state["profile"]
    c1, c2 = st.columns(2)
    with c1:
        p["name"] = st.text_input("Your name (optional)", value=p["name"])
        p["college_level"] = st.selectbox("Level", ["Undergrad", "Grad"], index=0 if p["college_level"] == "Undergrad" else 1)
        p["grade"] = st.text_input("Grade (optional)", value=p["grade"])
    with c2:
        p["intended_major"] = st.text_input("Intended major (optional)", value=p["intended_major"])
        p["word_limit"] = st.number_input("Target word limit", min_value=250, max_value=1000, value=int(p["word_limit"]), step=10)

    p["remember_words"] = st.text_input("Three words you want a reader to remember about you", value=p["remember_words"], placeholder="curious, resilient, builder")
    p["avoid_topics"] = st.text_area("What you want to avoid (topics, clichés, angles)", value=p["avoid_topics"], height=90)
    p["boundaries"] = st.text_area("Boundaries (topics you do not want to discuss)", value=p["boundaries"], height=70)

    st.info("Next: story mining. Write short bullets. No full sentences needed.")


MINING_QUESTIONS = {
    "A": [
        "A moment you realized you were wrong about something important",
        "A moment you felt proud, but not because of an award",
        "A moment you failed and what you did the next week",
        "A moment you chose discomfort on purpose",
        "A moment you had to speak up when it was easier to stay quiet",
    ],
    "B": [
        "What people assume about you that is inaccurate",
        "What you are unusually serious about",
        "What you are unexpectedly good at",
        "A belief you changed your mind about",
        "A contradiction in your life that shaped you",
    ],
    "C": [
        "A problem you naturally solve in groups",
        "What friends ask you for help with",
        "What you can do for hours without needing motivation",
        "What you notice that others miss",
        "A habit you built that changed your results",
    ],
    "D": [
        "A room you know too well and why",
        "A sound that instantly takes you back",
        "An object you keep for a weird reason",
        "A place where you feel most like yourself",
        "A routine you do that says something about you",
    ],
}


def section_mine():
    st.subheader("Step 2: Story mining")
    st.write("Add 1 to 3 bullets for each prompt. Short and concrete beats long and abstract.")

    mined = st.session_state["mined_ideas"]

    for b in ["A", "B", "C", "D"]:
        st.markdown(f"### {bucket_label(b)}")
        for q_i, q in enumerate(MINING_QUESTIONS[b], start=1):
            key = f"mine_{b}_{q_i}"
            existing = ""
            # Keep a lightweight input buffer for each prompt
            existing = st.session_state.get(key, "")
            txt = st.text_area(q, value=existing, key=key, height=80, placeholder="Write 1-3 bullets. Example: 'I taught my little brother math using drawings'")

        st.divider()

    if st.button("Save mined ideas"):
        # Collect all buffers into mined lists, one idea per line
        new_mined = {"A": [], "B": [], "C": [], "D": []}
        for b in ["A", "B", "C", "D"]:
            for q_i in range(1, len(MINING_QUESTIONS[b]) + 1):
                raw = st.session_state.get(f"mine_{b}_{q_i}", "") or ""
                lines = [safe_text(x).strip("-• \t") for x in raw.splitlines()]
                lines = [x for x in lines if x.strip()]
                new_mined[b].extend(lines)
        st.session_state["mined_ideas"] = new_mined

        # Create idea_details entries for each idea
        details = st.session_state["idea_details"]
        for b in ["A", "B", "C", "D"]:
            for i, idea in enumerate(new_mined[b], start=1):
                iid = idea_id(b, i)
                if iid not in details:
                    details[iid] = {"idea_text": idea}

        st.success("Saved. Next: choose a candidate and run the Scene Test.")

    st.caption("Tip: If everything feels generic, add a detail: place, object, dialogue, or a decision you made.")


def section_scene_test_and_meaning():
    st.subheader("Step 3: Scene Test and meaning")
    mined = st.session_state["mined_ideas"]
    all_ids = []
    for b in ["A", "B", "C", "D"]:
        for i, idea in enumerate(mined[b], start=1):
            all_ids.append((idea_id(b, i), idea))

    if not all_ids:
        st.warning("No mined ideas found. Go to Story mining and save your ideas.")
        return

    options = [format_candidate_title(idea, iid) for iid, idea in all_ids]
    selected_display = st.selectbox("Pick one idea to develop", options=options, index=0)
    selected_iid = re.search(r"\(([^)]+)\)\s*$", selected_display).group(1)  # type: ignore
    st.session_state["selected_idea_id"] = selected_iid

    d = st.session_state["idea_details"].setdefault(selected_iid, {})
    d["idea_text"] = next((idea for iid, idea in all_ids if iid == selected_iid), d.get("idea_text", ""))

    st.markdown("#### Scene Test")
    d["scene_where"] = st.text_area("Where are you, who is there, and what is happening (2-4 lines)", value=d.get("scene_where", ""), height=90)
    d["scene_tension"] = st.text_area("What is the tension or stake (what could go wrong and why it matters)", value=d.get("scene_tension", ""), height=90)
    d["scene_decision"] = st.text_area("What did you choose to do (action, decision, risk)", value=d.get("scene_decision", ""), height=90)
    d["scene_change"] = st.text_area("What changed afterward (in behavior, not just feelings)", value=d.get("scene_change", ""), height=90)

    st.markdown("#### Meaning without moralizing")
    d["belief_before"] = st.text_input("What did you believe before this moment", value=d.get("belief_before", ""))
    d["belief_now"] = st.text_input("What do you believe now", value=d.get("belief_now", ""))
    d["skill_built"] = st.text_input("What skill or habit did you build because of it", value=d.get("skill_built", ""))
    d["behavior_now"] = st.text_area("How do you act differently today (specific behaviors)", value=d.get("behavior_now", ""), height=80)

    values = [
        "curiosity", "courage", "empathy", "discipline", "initiative",
        "integrity", "leadership", "resilience", "humility", "creativity",
    ]
    chosen = st.multiselect("Pick 2 values this story proves", options=values, default=d.get("values_chosen", [])[:2])
    d["values_chosen"] = chosen[:2]

    st.caption("If you cannot answer the Scene Test, this topic may be too abstract. That is useful data. Choose another idea and compare.")


def section_rank():
    st.subheader("Step 4: Score and rank your best story options")

    mined = st.session_state["mined_ideas"]
    details = st.session_state["idea_details"]

    # Build list of all ideas
    all_ideas: List[Tuple[str, str]] = []
    for b in ["A", "B", "C", "D"]:
        for i, idea in enumerate(mined[b], start=1):
            all_ideas.append((idea_id(b, i), idea))

    if not all_ideas:
        st.warning("No mined ideas found. Go to Story mining and save your ideas.")
        return

    st.write("Score up to 5 ideas. You can change scores later.")
    pick_ids = [format_candidate_title(idea, iid) for iid, idea in all_ideas]
    chosen_displays = st.multiselect("Choose ideas to score (2-5)", options=pick_ids, default=pick_ids[: min(3, len(pick_ids))])
    chosen_iids = []
    for disp in chosen_displays:
        m = re.search(r"\(([^)]+)\)\s*$", disp)
        if m:
            chosen_iids.append(m.group(1))

    scored: List[Tuple[str, Dict[str, int], int]] = []

    for iid in chosen_iids:
        idea_text = details.get(iid, {}).get("idea_text", "")
        st.markdown(f"### {safe_text(idea_text)}  ({iid})")

        cols = st.columns(5)
        s = {}
        with cols[0]:
            s["specificity"] = st.slider("Specificity", 1, 5, 3, key=f"{iid}_spec", help="Can a reader see it as a scene?")
        with cols[1]:
            s["depth"] = st.slider("Depth", 1, 5, 3, key=f"{iid}_depth", help="Does it show internal change, not just events?")
        with cols[2]:
            s["agency"] = st.slider("Agency", 1, 5, 3, key=f"{iid}_agency", help="Are you driving actions and decisions?")
        with cols[3]:
            s["originality"] = st.slider("Originality", 1, 5, 3, key=f"{iid}_orig", help="Does it avoid common topics or common framing?")
        with cols[4]:
            s["values"] = st.slider("Values proof", 1, 5, 3, key=f"{iid}_values", help="Does it clearly prove 1-2 values with evidence?")
        total = sum(s.values())
        scored.append((iid, s, total))

    if st.button("Save ranking"):
        scored_sorted = sorted(scored, key=lambda x: x[2], reverse=True)
        st.session_state["top_candidates"] = scored_sorted
        if scored_sorted and not st.session_state["selected_idea_id"]:
            st.session_state["selected_idea_id"] = scored_sorted[0][0]
        st.success("Ranking saved. Next: pick your story and build your core argument.")

    if st.session_state["top_candidates"]:
        st.markdown("#### Current Top Candidates")
        for rank, (iid, s, total) in enumerate(st.session_state["top_candidates"], start=1):
            txt = details.get(iid, {}).get("idea_text", "")
            st.write(f"{rank}. {safe_text(txt)} ({iid}) | Total {total}/25")


def section_core_and_scene_bank():
    st.subheader("Step 5: Core argument and scene bank")

    selected_iid = st.session_state["selected_idea_id"]
    if not selected_iid:
        st.warning("Select a story first. Use Scene Test or Ranking.")
        return

    d = st.session_state["idea_details"].get(selected_iid, {})
    st.markdown(f"**Chosen story:** {safe_text(d.get('idea_text',''))} ({selected_iid})")

    st.write("Build a one-sentence spine. Keep it simple. No big moral statements.")
    template = st.selectbox(
        "Pick a template",
        [
            "I used to ___, but after ___, I now ___, which shows I am ___.",
            "The pattern in my life is ___: when I face ___, I ___, and I learned ___.",
            "I care about ___ because ___. I proved it when ___, and now I ___ differently.",
        ],
        index=0,
    )

    core = st.text_area("Write your core argument sentence", value=st.session_state["core_argument"], height=80, placeholder=template)
    st.session_state["core_argument"] = safe_text(core).strip()

    st.markdown("### Scene bank (4-6 scenes)")
    st.caption("Each scene should be short. The point is material and structure, not polished writing.")

    scenes = st.session_state["scene_bank"]
    max_scenes = 6
    min_scenes = 4

    def scene_editor(i: int):
        s = scenes[i]
        st.markdown(f"**Scene {i+1}**")
        s["what"] = st.text_area("What happened (2-3 lines)", value=s.get("what", ""), key=f"scene_what_{i}", height=80)
        s["reveals"] = st.text_input("What it reveals about you", value=s.get("reveals", ""), key=f"scene_rev_{i}")
        s["connects"] = st.text_input("How it connects to your core argument", value=s.get("connects", ""), key=f"scene_con_{i}")
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Move up", key=f"scene_up_{i}") and i > 0:
                scenes[i - 1], scenes[i] = scenes[i], scenes[i - 1]
                st.rerun()
        with c2:
            if st.button("Delete", key=f"scene_del_{i}"):
                scenes.pop(i)
                st.rerun()

    # Add default scenes if empty
    if not scenes:
        st.session_state["scene_bank"] = [{"what": "", "reveals": "", "connects": ""} for _ in range(min_scenes)]
        scenes = st.session_state["scene_bank"]

    for i in range(len(scenes)):
        scene_editor(i)
        st.divider()

    if len(scenes) < max_scenes:
        if st.button("Add another scene"):
            scenes.append({"what": "", "reveals": "", "connects": ""})
            st.rerun()

    st.info("Next: generate an outline from your core argument and scene bank.")


def section_outline_and_export():
    st.subheader("Step 6: Outline builder (no prose)")
    core = st.session_state["core_argument"]
    scenes = st.session_state["scene_bank"]

    if not core.strip():
        st.warning("Write your core argument first.")
        return

    if not scenes or all(not (s.get("what","").strip()) for s in scenes):
        st.warning("Add at least 2 scenes with real content.")
        return

    template = st.selectbox(
        "Choose an outline template",
        ["Classic Narrative Arc", "Montage", "Problem-Solver Arc"],
        index=["Classic Narrative Arc", "Montage", "Problem-Solver Arc"].index(st.session_state["outline_template"])
        if st.session_state["outline_template"] in ["Classic Narrative Arc", "Montage", "Problem-Solver Arc"]
        else 0,
    )
    st.session_state["outline_template"] = template

    if st.button("Generate outline"):
        st.session_state["outline"] = build_outline(template, core, scenes)

    if st.session_state["outline"]:
        st.text_area("Your outline", value=st.session_state["outline"], height=380)

    st.markdown("### Optional: paste your draft for coaching")
    limit = int(st.session_state["profile"].get("word_limit", 650))
    st.caption(f"Target word limit: {limit}. The coach will not rewrite your essay.")

    draft = st.text_area("Draft (optional)", value=st.session_state["draft_text"], height=220)
    st.session_state["draft_text"] = draft

    wc = word_count(draft)
    if draft.strip():
        delta = wc - limit
        st.write(f"Draft word count: **{wc}** (delta vs target: {delta:+d})")

    st.success("You can download your Personal Statement Packet from the sidebar at any time.")


def section_coach_chat():
    st.subheader("Coach chat (bottom bar style)")
    st.caption("This chat critiques, asks questions, and helps you plan. It will not write your essay.")

    chat = st.session_state["coach_chat"]

    # Show history
    for m in chat[-20:]:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    # Bottom chat bar
    prompt = st.chat_input("Ask for critique, clarity, cuts, or next questions")
    if prompt:
        user_msg = safe_text(prompt)
        chat.append({"role": "user", "content": user_msg})

        # Add context snapshot for better coaching (without writing essay)
        context = {
            "core_argument": st.session_state["core_argument"],
            "selected_story": st.session_state["idea_details"].get(st.session_state["selected_idea_id"], {}).get("idea_text", ""),
            "scene_test": {
                k: st.session_state["idea_details"].get(st.session_state["selected_idea_id"], {}).get(k, "")
                for k in ["scene_where", "scene_tension", "scene_decision", "scene_change"]
            },
            "values": st.session_state["idea_details"].get(st.session_state["selected_idea_id"], {}).get("values_chosen", []),
            "outline_template": st.session_state["outline_template"],
            "outline": st.session_state["outline"],
            "word_limit": st.session_state["profile"].get("word_limit", 650),
        }
        context_msg = safe_text(
            "Context snapshot (for coaching, do not rewrite the essay):\n"
            + json.dumps(context, ensure_ascii=False, indent=2)
        )

        messages_for_model = chat[-10:] + [{"role": "user", "content": context_msg}]

        # Generate coach response
        g = st.session_state["gemini"]
        if g["use_gemini"] and g["api_key"] and g["model_name"]:
            reply = gemini_coach(g["api_key"], g["model_name"], messages_for_model)
        else:
            reply = local_coach(messages_for_model)

        chat.append({"role": "assistant", "content": reply})
        st.rerun()

    # Quick actions
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        if st.button("Ask me what to cut"):
            chat.append({"role": "user", "content": "Tell me what to cut first and why. Use my outline or draft if present. Do not rewrite."})
            st.rerun()
    with c2:
        if st.button("Ask for stronger hook options"):
            chat.append({"role": "user", "content": "Give me 3 hook directions based on my scene bank. Only describe the hook move, do not write paragraphs."})
            st.rerun()
    with c3:
        if st.button("Clear chat"):
            st.session_state["coach_chat"] = []
            st.rerun()


def main():
    ensure_state()
    sidebar()

    nav = st.sidebar.radio(
        "Navigate",
        [
            "Start",
            "Story mining",
            "Scene test and meaning",
            "Ranking",
            "Core argument and scene bank",
            "Outline and export",
            "Coach chat",
        ],
        index=[
            "Start",
            "Story mining",
            "Scene test and meaning",
            "Ranking",
            "Core argument and scene bank",
            "Outline and export",
            "Coach chat",
        ].index(st.session_state.get("step", "Start")),
    )
    st.session_state["step"] = nav

    if nav == "Start":
        section_start()
    elif nav == "Story mining":
        section_mine()
    elif nav == "Scene test and meaning":
        section_scene_test_and_meaning()
    elif nav == "Ranking":
        section_rank()
    elif nav == "Core argument and scene bank":
        section_core_and_scene_bank()
    elif nav == "Outline and export":
        section_outline_and_export()
    elif nav == "Coach chat":
        section_coach_chat()


if __name__ == "__main__":
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    main()
