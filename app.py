import streamlit as st
import pandas as pd
import json
import random
import os

# Import Modules
from modules.database import save_session, get_all_sessions
from modules.nlp_engine import compute_similarity
from modules.speech_engine import speak, listen_audio

# --- CONFIG ---
st.set_page_config(page_title="SmartPrep Phase 1", layout="wide")

# --- SESSION STATE ---
if 'app_state' not in st.session_state: st.session_state.app_state = 'setup'
if 'question_queue' not in st.session_state: st.session_state.question_queue = []
if 'current_idx' not in st.session_state: st.session_state.current_idx = 0
if 'results_log' not in st.session_state: st.session_state.results_log = []

# --- HELPER: LOAD QUESTIONS ---
def get_interview_set(role):
    try:
        with open(os.path.join("data", "questions.json"), 'r') as f:
            data = json.load(f)
    except: return []
    
    role_qs = data.get(role, [])
    intro = [q for q in role_qs if q['type'] == 'Introduction']
    tech = [q for q in role_qs if q['type'] == 'Technical']
    
    # 1 Intro + 4 Technical
    sel_intro = random.sample(intro, 1) if intro else []
    sel_tech = random.sample(tech, min(4, len(tech))) 
    
    return sel_intro + sel_tech

# ==========================================
# PHASE 1: SETUP
# ==========================================
if st.session_state.app_state == 'setup':
    st.title("🤖 SmartPrep: Technical Interview")
    
    try:
        with open(os.path.join("data", "questions.json"), 'r') as f:
            roles = list(json.load(f).keys())
    except: roles = ["Python Backend Developer"]

    role = st.selectbox("Select Target Role", roles)
    
    if st.button("Start Interview"):
        qs = get_interview_set(role)
        if qs:
            st.session_state.question_queue = qs
            st.session_state.results_log = []
            st.session_state.app_state = 'interview'
            st.rerun()

# ==========================================
# PHASE 2: INTERVIEW LOOP
# ==========================================
elif st.session_state.app_state == 'interview':
    qs = st.session_state.question_queue
    idx = st.session_state.current_idx
    
    if idx >= len(qs):
        st.session_state.app_state = 'report'
        st.rerun()
    
    current_q = qs[idx]
    
    st.progress((idx+1)/len(qs), text=f"Question {idx+1} of {len(qs)}")
    st.markdown(f"### 🗣️ {current_q['text']}")
    
    # Auto-Speak
    if f"spoken_{idx}" not in st.session_state:
        speak(current_q['text'])
        st.session_state[f"spoken_{idx}"] = True
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("🎙️ Answer (Mic)")
        
        # RECORDING BUTTON
        if st.button("Start Recording"):
            with st.spinner("Listening... (You can pause for up to 3 seconds)"):
                text = listen_audio()
                
                if text:
                    st.success("Captured!")
                    st.session_state.temp_ans = text
                    st.rerun()
                else:
                    # SILENCE DETECTED
                    st.error("No audio detected.")
                    st.session_state.silence_detected = True

        # Handle Silence / "I don't know"
        if st.session_state.get('silence_detected'):
            st.warning("You stayed silent. What would you like to do?")
            c_retry, c_skip = st.columns(2)
            with c_retry:
                if st.button("Try Recording Again"):
                    st.session_state.silence_detected = False
                    st.rerun()
            with c_skip:
                if st.button("Skip Question (Score 0)"):
                    st.session_state.temp_ans = "Skipped / Don't Know"
                    st.session_state.silence_detected = False
                    st.rerun()

    with col2:
        st.info("⌨️ Answer (Text)")
        manual_ans = st.text_area("Type here if mic fails:")
        if st.button("Submit Text"):
            st.session_state.temp_ans = manual_ans
            st.rerun()

    # REVIEW & SUBMIT
    if 'temp_ans' in st.session_state:
        st.markdown("---")
        st.write(f"**Your Answer:** {st.session_state.temp_ans}")
        
        if st.button("✅ Confirm & Next"):
            # Grading Logic
            if current_q['type'] == 'Introduction':
                score = 100
                feedback = "Good introduction! Confidence is key here."
            elif "Skipped" in st.session_state.temp_ans:
                score = 0
                feedback = "Question skipped."
            else:
                score, feedback = compute_similarity(st.session_state.temp_ans, current_q['model_answer'])
            
            # Save Data
            st.session_state.results_log.append({
                "question": current_q['text'],
                "answer": st.session_state.temp_ans,
                "score": score,
                "feedback": feedback
            })
            
            save_session(current_q['type'], current_q['text'], st.session_state.temp_ans, score, feedback, "Good", "Good")
            
            # Cleanup
            del st.session_state['temp_ans']
            if 'silence_detected' in st.session_state: del st.session_state['silence_detected']
            
            st.session_state.current_idx += 1
            st.rerun()

# ==========================================
# PHASE 3: REPORT
# ==========================================
elif st.session_state.app_state == 'report':
    st.title("📊 Interview Report")
    st.balloons()
    
    results = st.session_state.results_log
    
    # Calculate Average (Excluding Intro if you prefer, or keeping it)
    # Here we include everything for simplicity
    if results:
        avg_score = sum(r['score'] for r in results)/len(results)
    else:
        avg_score = 0
    
    m1, m2 = st.columns(2)
    m1.metric("Overall Score", f"{int(avg_score)}%")
    m2.metric("Questions Answered", len(results))
    
    st.divider()
    for i, res in enumerate(results):
        with st.expander(f"Q{i+1}: {res['question']} (Score: {res['score']}%)"):
            st.write(f"**Answer:** {res['answer']}")
            st.info(f"**Feedback:** {res['feedback']}")
            
    if st.button("Start New Session"):
        st.session_state.app_state = 'setup'
        st.session_state.current_idx = 0
        st.rerun()