import os
import re
import time
from typing import Dict, List, Optional, Tuple

import streamlit as st

# Gemini SDK
import google.generativeai as genai


# ---------------------------
# Core configuration
# ---------------------------

APP_TITLE = "Common App Personal Statement Coach"
APP_TAGLINE = "A step-by-step coach to help you discover your story, connect the dots, and write your own essay."

# Try models in order. We pick fast, widely available defaults first.
MODEL_CANDIDATES = [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-2.0-flash",
]

DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_OUTPUT_TOKENS = 1400

# User preference: no em dashes in output
EM_DASH = "—"


# ---------------------------
# Utilities
# ---------------------------

def clean_no_emdash(text: str) -> str:
    """Replace em dashes with a normal hyphen and normalize spacing."""
    if not text:
        return text
    text = text.replace(EM_DASH, "-")
    # Also handle common Unicode long dashes
    text = text.replace("\u2014", "-").replace("\u2013", "-")
    # Light cleanup: avoid triple spaces
    text = re.sub(r"[ \t]{3,}", "  ", text)
    return text.strip()


def get_api_key() -> Optional[str]:
    """Read key from Streamlit secrets first, then env var."""
    key = None
    try:
        key = st.secrets.get("GOOGLE_API_KEY", None)
    except Exception:
        key = None
    if not key:
        key = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY".lower())
    return key


def init_gemini() -> Tuple[Optional[genai.GenerativeModel], Optional[str]]:
    """Initialize Gemini model. Returns (model, model_name_used)."""
    api_key = get_api_key()
    if not api_key:
        return None, None

    genai.configure(api_key=api_key)

    # Create a system instruction that keeps the app aligned:
    # guidance-only, no writing whole essay, ask questions, give frameworks.
    system_instruction = clean_no_emdash(
        """
You are a Common App personal statement coach.
You do NOT write the student's essay for them.
You guide them to discover a strong topic, build a narrative arc, and write in their own voice.
You use questions, structured exercises, and clear options.
You avoid generic morals. You avoid cliches. You avoid em dashes.
You help the student connect:
- specific moments
- internal change
- values
- habits
- future direction
You are direct, practical, and specific.
If a student asks you to write the full essay, you refuse and instead provide:
- an outline
- a paragraph plan
- sentence-level examples they can adapt
- a revision checklist
Keep suggestions grounded in what the student has actually shared.
        """.strip()
    )

    # Try models in order until one works.
    last_error = None
    for name in MODEL_CANDIDATES:
        try:
            model = genai.GenerativeModel(
                model_name=name,
                system_instruction=system_instruction,
            )
            # Simple dry run prompt to ensure model works
            _ = model.generate_content("Reply with: OK")
            return model, name
        except Exception as e:
            last_error = str(e)

    # If all fail, show last error in UI later
    st.session_state["gemini_init_error"] = last_error
    return None, None


def gemini_generate(model: genai.GenerativeModel, prompt: str, temperature: float = DEFAULT_TEMPERATURE) -> str:
    """Single Gemini call wrapper with safe defaults."""
    resp = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
        ),
    )
    text = getattr(resp, "text", "") or ""
    return clean_no_emdash(text)


def word_count(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"\b\w+\b", text))


def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


# ---------------------------
# Coaching flow logic
# ---------------------------

FLOW_STEPS = [
    "Setup",
    "Story Mining",
    "Themes and Values",
    "Angles and Thesis",
    "Outline Builder",
    "Draft Coaching",
    "Revision Plan",
]

def default_state():
    return {
        "step": FLOW_STEPS[0],
        "profile": {
            "grade_level": "Undergrad (Common App)",
            "prompt": "Common App Personal Statement (choose your own topic)",
            "word_limit": 650,
            "tone_pref": "Authentic, specific, not over-polished",
            "risk_flags": [],
        },
        "mining": {
            "moments_raw": "",
            "constraints_raw": "",
            "people_places_raw": "",
            "turning_points_raw": "",
        },
        "themes": {
            "values_raw": "",
            "patterns_raw": "",
            "skills_raw": "",
            "future_raw": "",
        },
        "angles": {
            "angle_choice": "",
            "thesis": "",
            "throughline": "",
            "what_changed": "",
        },
        "outline": {
            "outline_text": "",
            "scene_bank": "",
            "details_to_show": "",
        },
        "draft": {
            "draft_text": "",
            "coach_notes": "",
            "paragraph_plan": "",
        },
        "revision": {
            "revision_checklist": "",
            "next_actions": "",
        },
        "artifacts": {
            "latest_suggestion_title": "",
            "latest_suggestion_body": "",
        },
        "chat": {
            "messages": [],  # list of dicts {role, content}
            "suggested_questions": [],
            "pending_user_message": None,
        },
    }


def set_step(step_name: str):
    if step_name in FLOW_STEPS:
        st.session_state["app"]["step"] = step_name


def add_chat(role: str, content: str):
    st.session_state["app"]["chat"]["messages"].append({"role": role, "content": clean_no_emdash(content)})


def build_suggested_questions(step_name: str, app_state: Dict) -> List[str]:
    """Suggested question chips shown above the chat bar."""
    p = app_state["profile"]
    mining = app_state["mining"]
    angles = app_state["angles"]
    draft = app_state["draft"]

    if step_name == "Setup":
        return [
            "What makes a Common App personal statement actually memorable?",
            "How do I pick a topic if my life feels normal?",
            "What should I avoid so my essay does not sound generic?",
            "How do I show growth without preaching a lesson?",
        ]

    if step_name == "Story Mining":
        return [
            "Give me 3 more questions to help me find better story moments.",
            "Which of my moments is strongest and why?",
            "What details should I include to make it vivid?",
            "How do I avoid trauma dumping but still be honest?",
        ]

    if step_name == "Themes and Values":
        return [
            "What values do my moments reveal and how can I prove them?",
            "Help me connect my story to a future direction without forcing it.",
            "How can I show personality on the page without being cringe?",
            "How do I turn a skill into a story, not a resume line?",
        ]

    if step_name == "Angles and Thesis":
        angle_hint = angles.get("angle_choice") or "my angle"
        return [
            f"Is {angle_hint} too common? How do I make it unique?",
            "Write 3 possible one sentence throughlines for my essay.",
            "What is the strongest change in me that the reader can feel?",
            "What is one risky but smart creative structure I could try?",
        ]

    if step_name == "Outline Builder":
        return [
            "Is my outline too summary-heavy? Where should I add scenes?",
            "What should be the first 3 sentences of my hook, in my voice?",
            "Which paragraph is the weak link and why?",
            "How do I end without sounding like a motivational speech?",
        ]

    if step_name == "Draft Coaching":
        wc = word_count(draft.get("draft_text", ""))
        lim = p.get("word_limit", 650)
        return [
            f"My draft is {wc} words. How do I trim to {lim} without losing meaning?",
            "Point out where I am telling instead of showing.",
            "Give me line edits for clarity and voice, but do not rewrite the whole essay.",
            "What is missing that would make this feel more personal and specific?",
        ]

    if step_name == "Revision Plan":
        return [
            "Give me a 3 day revision schedule.",
            "What are the top 5 fixes that will move the needle most?",
            "How do I sanity check my essay for authenticity?",
            "Can you help me craft a short reflection I can keep off the essay, but use to revise?",
        ]

    return [
        "What should I do next?",
        "What is my biggest weakness right now?",
        "Give me a concrete example of how to improve one paragraph.",
        "Help me choose between two ideas.",
    ]


def coaching_prompt(step_name: str, app_state: Dict) -> str:
    """Build a strong instruction prompt for the coach, step-specific."""
    p = app_state["profile"]
    mining = app_state["mining"]
    themes = app_state["themes"]
    angles = app_state["angles"]
    outline = app_state["outline"]
    draft = app_state["draft"]

    # Compact context for the model
    context = f"""
Student context:
- Application: {p['grade_level']}
- Essay: {p['prompt']}
- Word limit: {p['word_limit']}
- Tone preference: {p['tone_pref']}

Story mining notes:
- Moments: {mining.get('moments_raw','').strip()}
- Constraints: {mining.get('constraints_raw','').strip()}
- People and places: {mining.get('people_places_raw','').strip()}
- Turning points: {mining.get('turning_points_raw','').strip()}

Themes and values:
- Values: {themes.get('values_raw','').strip()}
- Patterns: {themes.get('patterns_raw','').strip()}
- Skills: {themes.get('skills_raw','').strip()}
- Future direction: {themes.get('future_raw','').strip()}

Angles and throughline:
- Angle choice: {angles.get('angle_choice','').strip()}
- Thesis: {angles.get('thesis','').strip()}
- Throughline: {angles.get('throughline','').strip()}
- What changed: {angles.get('what_changed','').strip()}

Outline:
- Outline: {outline.get('outline_text','').strip()}
- Scene bank: {outline.get('scene_bank','').strip()}
- Details to show: {outline.get('details_to_show','').strip()}

Draft:
- Draft text: {draft.get('draft_text','').strip()}
    """.strip()

    # Step-specific instructions
    if step_name == "Setup":
        task = """
Task:
Help the student set up the work correctly.
1) Explain what a strong personal statement does in 4 to 6 crisp bullets.
2) Give 3 topic selection rules and 3 topic selection traps.
3) Ask exactly 8 targeted questions to learn what the student could write about.
Do not write an essay draft.
        """.strip()

    elif step_name == "Story Mining":
        task = """
Task:
Turn raw life material into usable story candidates.
1) Identify up to 4 strongest story moments from what they shared.
2) For each moment, give: why it works, what it reveals, and what scene details to show.
3) Ask 6 follow up questions that will make the moments more specific.
4) Recommend the best 1 to pursue and say why.
Do not write the essay.
        """.strip()

    elif step_name == "Themes and Values":
        task = """
Task:
Extract themes and values without forcing a moral.
1) List 5 to 8 values or traits that can be proven with scenes, not claims.
2) Map each value to an evidence moment (what to show).
3) Give 3 ways to connect the story to future direction without sounding resume-like.
4) Ask 5 questions to clarify what matters most to the student.
Do not write the essay.
        """.strip()

    elif step_name == "Angles and Thesis":
        task = """
Task:
Help the student find a unique angle and a clean throughline.
1) Propose 3 distinct angles for the same material (structure or lens changes).
2) For each angle, give a one sentence throughline and the emotional arc.
3) Recommend one angle and explain why it will be memorable.
4) Give a short thesis template the student can fill in.
Do not write the essay.
        """.strip()

    elif step_name == "Outline Builder":
        task = """
Task:
Build an outline that is scene-driven and specific.
1) Create a paragraph plan for a 650 word essay: hook, context, 2 to 3 scenes, reflection beats, ending.
2) For each paragraph: what happens, what it reveals, one detail to show.
3) Provide 6 possible opening hooks that are specific, not poetic.
4) Provide 4 ending strategies that avoid preaching.
Do not write the essay.
        """.strip()

    elif step_name == "Draft Coaching":
        task = """
Task:
Coach the student's draft without rewriting it for them.
1) Diagnose the top 5 issues: specificity, voice, structure, show vs tell, reflection depth, etc.
2) Give line-level guidance in small chunks: choose 6 to 10 sentences and show how to revise them with options, but do not rewrite the whole essay.
3) Identify 3 places to add a scene detail and 2 places to cut summary.
4) Give a trim plan to meet the word limit.
Be direct and practical.
        """.strip()

    else:  # Revision Plan
        task = """
Task:
Create a revision plan the student can execute.
1) Give a prioritized checklist (10 to 14 items) with quick tests.
2) Give a 3 session revision schedule: session goals and what to change.
3) Provide a final self-check rubric with scoring 1 to 5 for: voice, clarity, specificity, depth, coherence.
Do not write the essay.
        """.strip()

    final = f"{context}\n\n{task}"
    return clean_no_emdash(final)


def generate_step_suggestion(model: genai.GenerativeModel, step_name: str, app_state: Dict) -> str:
    prompt = coaching_prompt(step_name, app_state)
    return gemini_generate(model, prompt, temperature=0.65)


def ensure_chat_suggestions(app_state: Dict):
    step_name = app_state["step"]
    app_state["chat"]["suggested_questions"] = build_suggested_questions(step_name, app_state)


# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="wide")

if "app" not in st.session_state:
    st.session_state["app"] = default_state()

# Initialize Gemini once
if "gemini_model_name" not in st.session_state:
    model, used = init_gemini()
    st.session_state["gemini_model"] = model
    st.session_state["gemini_model_name"] = used

app = st.session_state["app"]
model = st.session_state.get("gemini_model", None)

st.title(APP_TITLE)
st.caption(APP_TAGLINE)

# Key status (no input UI)
api_key_present = bool(get_api_key())
colA, colB, colC = st.columns([1, 1, 2])
with colA:
    st.metric("Gemini", "Enabled" if (api_key_present and model) else "Not connected")
with colB:
    st.metric("Model", st.session_state.get("gemini_model_name") or "None")
with colC:
    err = st.session_state.get("gemini_init_error", "")
    if (api_key_present and not model) and err:
        st.warning(clean_no_emdash(f"Gemini could not initialize. Error: {err}"))

st.divider()

# Step navigation
nav_cols = st.columns(len(FLOW_STEPS))
for i, sname in enumerate(FLOW_STEPS):
    with nav_cols[i]:
        active = (app["step"] == sname)
        if st.button(("✅ " if active else "") + sname, use_container_width=True):
            set_step(sname)
            ensure_chat_suggestions(app)

st.divider()

left, right = st.columns([1, 1])

with left:
    st.subheader("Your inputs")

    # Setup step inputs
    if app["step"] == "Setup":
        app["profile"]["grade_level"] = st.selectbox(
            "Application level",
            ["Undergrad (Common App)", "Grad (Personal Statement or Statement of Purpose)"],
            index=0 if app["profile"]["grade_level"].startswith("Undergrad") else 1,
        )

        app["profile"]["prompt"] = st.text_input(
            "Essay question or prompt",
            value=app["profile"]["prompt"],
            help="For Common App, you can paste the exact prompt or keep it general.",
        )

        app["profile"]["word_limit"] = st.number_input(
            "Word limit",
            min_value=250,
            max_value=1200,
            value=int(app["profile"]["word_limit"]),
            step=10,
        )

        app["profile"]["tone_pref"] = st.text_input(
            "Tone preference",
            value=app["profile"]["tone_pref"],
        )

        st.info("Next: Story Mining. You will list moments. The coach will help you pick the best topic.")

    # Story mining inputs
    elif app["step"] == "Story Mining":
        app["mining"]["moments_raw"] = st.text_area(
            "List 6 to 12 specific moments from your life",
            value=app["mining"]["moments_raw"],
            height=220,
            help="Use concrete moments. A conversation, a failure, a small habit, a decision, a conflict, a win.",
        )
        app["mining"]["turning_points_raw"] = st.text_area(
            "Turning points and inflection moments",
            value=app["mining"]["turning_points_raw"],
            height=140,
        )
        app["mining"]["people_places_raw"] = st.text_area(
            "People and places that shaped you",
            value=app["mining"]["people_places_raw"],
            height=140,
        )
        app["mining"]["constraints_raw"] = st.text_area(
            "Constraints or context",
            value=app["mining"]["constraints_raw"],
            height=120,
            help="Anything the reader needs to understand your situation, without overexplaining.",
        )

    # Themes inputs
    elif app["step"] == "Themes and Values":
        app["themes"]["values_raw"] = st.text_area(
            "Values you think you have (even if unsure)",
            value=app["themes"]["values_raw"],
            height=120,
        )
        app["themes"]["patterns_raw"] = st.text_area(
            "Patterns in your life",
            value=app["themes"]["patterns_raw"],
            height=120,
            help="Repeated behaviors, choices, obsessions, responsibilities, interests.",
        )
        app["themes"]["skills_raw"] = st.text_area(
            "Skills that show up in your moments",
            value=app["themes"]["skills_raw"],
            height=120,
            help="Not a resume list. Skills that are proven by what you did.",
        )
        app["themes"]["future_raw"] = st.text_area(
            "Future direction",
            value=app["themes"]["future_raw"],
            height=120,
            help="What you care about next. Not necessarily your major.",
        )

    # Angles inputs
    elif app["step"] == "Angles and Thesis":
        app["angles"]["angle_choice"] = st.text_input(
            "Angle you are leaning toward",
            value=app["angles"]["angle_choice"],
            help="Example: 'I learned to lead through building systems' or 'I became calmer by learning to name my emotions'.",
        )
        app["angles"]["throughline"] = st.text_input(
            "One sentence throughline (rough is fine)",
            value=app["angles"]["throughline"],
        )
        app["angles"]["what_changed"] = st.text_area(
            "What changed in you, specifically",
            value=app["angles"]["what_changed"],
            height=120,
            help="A before and after, but shown with behaviors and choices.",
        )
        app["angles"]["thesis"] = st.text_area(
            "Thesis or core claim",
            value=app["angles"]["thesis"],
            height=120,
            help="One claim the whole essay proves with scenes.",
        )

    # Outline inputs
    elif app["step"] == "Outline Builder":
        app["outline"]["scene_bank"] = st.text_area(
            "Scene bank (3 to 6 scenes you can write)",
            value=app["outline"]["scene_bank"],
            height=140,
        )
        app["outline"]["details_to_show"] = st.text_area(
            "Details to show (sensory, dialogue, specific actions)",
            value=app["outline"]["details_to_show"],
            height=140,
        )
        app["outline"]["outline_text"] = st.text_area(
            "Your current outline (optional)",
            value=app["outline"]["outline_text"],
            height=180,
        )

    # Draft coaching inputs
    elif app["step"] == "Draft Coaching":
        app["draft"]["draft_text"] = st.text_area(
            "Paste your draft",
            value=app["draft"]["draft_text"],
            height=320,
        )
        st.caption(f"Word count: {word_count(app['draft']['draft_text'])} / {app['profile']['word_limit']}")

    # Revision plan inputs
    elif app["step"] == "Revision Plan":
        st.write("You can generate a revision plan even if you do not have a full draft yet.")
        if st.button("Copy outline into draft field", use_container_width=True):
            app["draft"]["draft_text"] = app["outline"].get("outline_text", "")

    st.divider()

    # Generate suggestion
    can_use_ai = bool(model)
    btn_label = "Generate coach guidance for this step"
    if st.button(btn_label, type="primary", use_container_width=True, disabled=not can_use_ai):
        if not can_use_ai:
            st.error("Gemini is not connected. Add GOOGLE_API_KEY in Streamlit secrets.")
        else:
            with st.spinner("Thinking..."):
                suggestion = generate_step_suggestion(model, app["step"], app)
            app["artifacts"]["latest_suggestion_title"] = f"Coach guidance: {app['step']}"
            app["artifacts"]["latest_suggestion_body"] = suggestion
            add_chat("assistant", f"I generated guidance for {app['step']}. Ask me anything about it.")
            ensure_chat_suggestions(app)

with right:
    st.subheader("Coach output")

    title = app["artifacts"].get("latest_suggestion_title", "")
    body = app["artifacts"].get("latest_suggestion_body", "")

    if title:
        st.markdown(f"### {title}")
    if body:
        st.write(body)
    else:
        st.info("Generate guidance on the left to see step-by-step coaching here.")

    st.divider()

    # Always show chat under the output
    st.subheader("Ask the coach")

    ensure_chat_suggestions(app)

    # Suggested questions row
    suggestions = app["chat"].get("suggested_questions", [])[:6]
    if suggestions:
        st.caption("Not sure what to ask? Try one of these:")
        chip_cols = st.columns(len(suggestions))
        for i, q in enumerate(suggestions):
            with chip_cols[i]:
                if st.button(q, use_container_width=True):
                    # Treat chip click as a user message
                    app["chat"]["pending_user_message"] = q

    # Render chat history
    for m in app["chat"]["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # If a chip set a pending message, process it automatically
    pending = app["chat"].get("pending_user_message", None)
    if pending:
        app["chat"]["pending_user_message"] = None
        add_chat("user", pending)
        if model:
            with st.spinner("Coach is replying..."):
                # Build a chat prompt with short context: last suggestion + step + last messages
                last_guidance = app["artifacts"].get("latest_suggestion_body", "")
                convo_tail = app["chat"]["messages"][-8:]
                convo_text = "\n".join([f"{x['role'].upper()}: {x['content']}" for x in convo_tail])

                prompt = clean_no_emdash(
                    f"""
You are continuing a coaching chat.
Current step: {app['step']}

Latest coach guidance:
{last_guidance}

Recent conversation:
{convo_text}

Now answer the user's latest message with:
- direct answer
- 2 to 4 concrete next actions
- 2 follow up questions that clarify the story
Do not write the full essay.
                    """.strip()
                )
                reply = gemini_generate(model, prompt, temperature=0.6)
            add_chat("assistant", reply)
        else:
            add_chat("assistant", "Gemini is not connected. Add GOOGLE_API_KEY in Streamlit secrets.")

        st.rerun()

    # Chat input bar
    user_msg = st.chat_input("Ask a question about the guidance, your story, or your outline")
    if user_msg:
        add_chat("user", user_msg)
        if model:
            with st.spinner("Coach is replying..."):
                last_guidance = app["artifacts"].get("latest_suggestion_body", "")
                convo_tail = app["chat"]["messages"][-10:]
                convo_text = "\n".join([f"{x['role'].upper()}: {x['content']}" for x in convo_tail])

                prompt = clean_no_emdash(
                    f"""
You are continuing a coaching chat.
Current step: {app['step']}

Latest coach guidance:
{last_guidance}

Recent conversation:
{convo_text}

Now answer the user's latest message with:
- direct answer
- 2 to 4 concrete next actions
- 2 follow up questions that clarify the story
Do not write the full essay.
                    """.strip()
                )
                reply = gemini_generate(model, prompt, temperature=0.6)
            add_chat("assistant", reply)
        else:
            add_chat("assistant", "Gemini is not connected. Add GOOGLE_API_KEY in Streamlit secrets.")
        st.rerun()
