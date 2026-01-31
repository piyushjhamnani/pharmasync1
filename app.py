import streamlit as st
import pandas as pd
import json
import random
import os
import cv2
import mediapipe as mp
import time
import threading
from PIL import Image
import numpy as np
from datetime import datetime

from modules.database import save_session
from modules.nlp_engine import compute_similarity
from modules.speech_engine import speak, listen_audio

# ================= CONFIG =================
st.set_page_config(page_title="SmartPrep Phase 1", layout="wide")

# ================= SAFE SESSION INIT =================
def initialize_session_state():
    defaults = {
        "app_state": "setup",
        "question_queue": [],
        "current_idx": 0,
        "results_log": [],
        "temp_ans": "",
        "last_audio_played": -1,
        "face_verified": False,
        "interview_started": False,
        "face_monitoring": False,
        "face_warnings": 0,
        "last_face_detection": 0,
        "face_detection_log": [],
        "camera_session": None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# Initialize MediaPipe
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

# ================= PERSISTENT FACE MONITORING =================
class FaceMonitor:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.face_detection = None
        self.cap = None
        self.running = False
        self.last_frame = None
        self.face_detected = False
        self.last_update_time = 0
        
    def start(self):
        """Start the face monitoring"""
        if self.running:
            return True
            
        try:
            # Initialize camera
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                st.error("Cannot open camera")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Initialize MediaPipe face detection
            self.face_detection = mp_face.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.5
            )
            
            self.running = True
            self.last_update_time = time.time()
            return True
            
        except Exception as e:
            st.error(f"Error starting camera: {e}")
            return False
    
    def stop(self):
        """Stop the face monitoring"""
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.face_detection:
            self.face_detection.close()
            self.face_detection = None
    
    def get_status(self):
        """Get current face detection status"""
        return {
            "detected": self.face_detected,
            "running": self.running,
            "last_update": self.last_update_time
        }
    
    def check_face(self):
        """Check for face in a single frame"""
        if not self.running or not self.cap:
            return False
            
        try:
            ret, frame = self.cap.read()
            if not ret:
                return False
            
            # Store last frame
            self.last_frame = frame.copy()
            
            # Try MediaPipe first
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_frame)
            
            if results and results.detections:
                self.face_detected = True
                self.last_update_time = time.time()
                
                # Draw detections on frame
                for detection in results.detections:
                    mp_draw.draw_detection(frame, detection)
                
                # Add timestamp and status
                cv2.putText(frame, f"✅ Face Detected - {datetime.now().strftime('%H:%M:%S')}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                self.last_frame = frame
                return True
            
            # Fallback to OpenCV
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            if len(faces) > 0:
                self.face_detected = True
                self.last_update_time = time.time()
                
                # Draw rectangles on frame
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Add timestamp and status
                cv2.putText(frame, f"✅ Face Detected - {datetime.now().strftime('%H:%M:%S')}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                self.last_frame = frame
                return True
            
            # No face detected
            self.face_detected = False
            cv2.putText(frame, f"❌ No Face - {datetime.now().strftime('%H:%M:%S')}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            self.last_frame = frame
            return False
            
        except Exception as e:
            print(f"Face check error: {e}")
            return False
    
    def get_frame_image(self):
        """Get the last frame as an image"""
        if self.last_frame is not None:
            return self.last_frame
        return None

# Create global face monitor
face_monitor = FaceMonitor()

# ================= SIMPLE FACE VERIFICATION =================
def simple_face_verification():
    """Simple one-time face verification"""
    st.markdown("### 🔒 Face Verification")
    
    if st.session_state.face_verified:
        st.success("✅ Face already verified!")
        return True
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📸 Check Face Now", type="primary", key="check_face"):
            # Start face monitor temporarily
            if face_monitor.start():
                with st.spinner("Checking for face..."):
                    # Check for 3 seconds
                    face_count = 0
                    total_checks = 6
                    
                    for i in range(total_checks):
                        if face_monitor.check_face():
                            face_count += 1
                        time.sleep(0.5)  # Check every 0.5 seconds
                    
                    # Show result
                    if face_count >= 3:  # At least 50% detection
                        st.session_state.face_verified = True
                        st.success(f"✅ Face verified! ({face_count}/{total_checks} checks)")
                        
                        # Show the last frame
                        frame = face_monitor.get_frame_image()
                        if frame is not None:
                            st.image(frame, channels="BGR", caption="Last captured frame")
                    else:
                        st.error(f"❌ Face not detected properly. ({face_count}/{total_checks} checks)")
                        
                        # Show the last frame
                        frame = face_monitor.get_frame_image()
                        if frame is not None:
                            st.image(frame, channels="BGR", caption="Last captured frame")
                    
                    # Stop the monitor
                    face_monitor.stop()
                
                st.rerun()
            else:
                st.error("Could not access camera")
    
    with col2:
        if st.button("⏭️ Skip Face Check", type="secondary"):
            st.session_state.face_verified = True
            st.warning("⚠️ Face verification skipped")
            st.rerun()
    
    if not st.session_state.face_verified:
        st.info("Please click 'Check Face Now' to verify your presence before starting the interview.")
    
    return st.session_state.face_verified

# ================= INTERVIEW FACE MONITORING =================
def start_continuous_monitoring():
    """Start continuous face monitoring for interview"""
    if not face_monitor.running:
        return face_monitor.start()
    return True

def stop_continuous_monitoring():
    """Stop continuous face monitoring"""
    face_monitor.stop()

def show_face_monitor_during_interview():
    """Show face monitoring during interview"""
    st.markdown("### 👁️ Continuous Face Monitoring")
    
    # Status indicators
    col1, col2 = st.columns(2)
    
    with col1:
        if face_monitor.running:
            st.success("📹 Camera: ACTIVE")
        else:
            st.error("📹 Camera: INACTIVE")
    
    with col2:
        status = face_monitor.get_status()
        if status["detected"]:
            st.success("👤 Face: DETECTED")
            time_since = time.time() - status["last_update"]
            if time_since < 2:
                st.caption("✅ Recently seen")
            else:
                st.warning(f"⚠️ Last seen {int(time_since)}s ago")
        else:
            st.error("👤 Face: NOT DETECTED")
    
    # Camera feed placeholder
    cam_placeholder = st.empty()
    
    # Check face periodically
    if face_monitor.running:
        # Perform a face check
        face_detected = face_monitor.check_face()
        
        # Update session state
        st.session_state.last_face_detection = time.time() if face_detected else st.session_state.last_face_detection
        
        # Log detection
        if len(st.session_state.face_detection_log) < 1000:  # Prevent memory issues
            st.session_state.face_detection_log.append({
                "timestamp": time.time(),
                "detected": face_detected
            })
        
        # Show camera feed
        frame = face_monitor.get_frame_image()
        if frame is not None:
            cam_placeholder.image(frame, channels="BGR", use_column_width=True)
        
        # Check for prolonged absence
        if not face_detected and face_monitor.running:
            time_since_last = time.time() - st.session_state.last_face_detection
            if time_since_last > 5:  # 5 seconds without face
                st.warning("⚠️ Face not detected for 5+ seconds. Please stay in frame!")
                st.session_state.face_warnings += 1
    else:
        cam_placeholder.info("Camera monitoring will start when you begin answering questions")
    
    return face_monitor.running

# ================= QUESTIONS =================
def get_interview_set(role):
    try:
        questions_path = os.path.join("data", "questions.json")
        if not os.path.exists(questions_path):
            # Create sample questions if file doesn't exist
            sample_questions = {
                "Python Backend Developer": [
                    {
                        "type": "Introduction",
                        "text": "Tell me about yourself.",
                        "model_answer": "I am a Python developer with experience in backend systems, APIs, and databases."
                    },
                    {
                        "type": "Technical",
                        "text": "What is the difference between list and tuple in Python?",
                        "model_answer": "Lists are mutable while tuples are immutable. Lists use square brackets [] and tuples use parentheses ()."
                    },
                    {
                        "type": "Technical",
                        "text": "What are Python decorators?",
                        "model_answer": "Decorators are functions that modify the behavior of other functions. They use the @ symbol."
                    },
                    {
                        "type": "Technical",
                        "text": "Explain the GIL in Python.",
                        "model_answer": "GIL stands for Global Interpreter Lock. It allows only one thread to execute Python bytecode at a time."
                    },
                    {
                        "type": "Technical",
                        "text": "What is Django ORM?",
                        "model_answer": "Django ORM is an Object-Relational Mapping layer that allows interacting with databases using Python objects."
                    }
                ]
            }
            os.makedirs("data", exist_ok=True)
            with open(questions_path, 'w') as f:
                json.dump(sample_questions, f, indent=2)
        
        with open(questions_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        st.error(f"Error loading questions: {e}")
        return []
    
    if role not in data:
        available_roles = list(data.keys())
        if available_roles:
            role = available_roles[0]
        else:
            return []
    
    role_qs = data.get(role, [])
    
    if not role_qs:
        return []
    
    intro = [q for q in role_qs if q.get('type') == 'Introduction']
    tech = [q for q in role_qs if q.get('type') == 'Technical']
    
    sel_intro = random.sample(intro, min(1, len(intro))) if intro else []
    sel_tech = random.sample(tech, min(4, len(tech))) if tech else []
    
    return sel_intro + sel_tech

# ================= AUTOMATIC AUDIO PLAYBACK =================
def play_question_audio_automatically():
    """Automatically play audio for current question if not already played"""
    try:
        current_idx = st.session_state.current_idx
        if current_idx != st.session_state.last_audio_played:
            qs = st.session_state.question_queue
            if current_idx < len(qs):
                question_text = qs[current_idx]['text']
                speak(question_text)
                st.session_state.last_audio_played = current_idx
    except Exception as e:
        # Don't show error, just continue
        pass

# ================= ANSWER PROCESSING =================
def process_answer(question, answer_text):
    """Process and score an answer"""
    try:
        if question.get('type') == 'Introduction':
            score = 100
            feedback = "Good introduction!"
        else:
            score, feedback = compute_similarity(
                answer_text,
                question.get('model_answer', '')
            )
        
        # Get face monitoring stats for this answer
        face_detection_during_answer = []
        recent_logs = st.session_state.face_detection_log[-10:]  # Last 10 checks
        if recent_logs:
            detection_rate = sum(1 for log in recent_logs if log["detected"]) / len(recent_logs) * 100
        else:
            detection_rate = 0
        
        st.session_state.results_log.append({
            "question": question['text'],
            "answer": answer_text,
            "score": score,
            "feedback": feedback,
            "face_detection_rate": detection_rate,
            "face_warnings": st.session_state.face_warnings
        })
        
        # Save to database
        try:
            save_session(
                question_type=question.get('type', 'Technical'),
                question_text=question['text'],
                answer=answer_text,
                score=score,
                feedback=feedback,
                posture="Good" if detection_rate > 70 else "Needs improvement",
                eye_contact="Good" if detection_rate > 70 else "Limited"
            )
        except Exception as e:
            # Continue even if database save fails
            pass
        
        # Reset and move to next question
        st.session_state.temp_ans = ""
        st.session_state.current_idx += 1
        st.rerun()
    
    except Exception as e:
        st.error(f"Error processing answer: {e}")

# =====================================================
# SETUP PAGE
# =====================================================
if st.session_state.app_state == 'setup':
    st.title("🤖 SmartPrep Interview Practice")
    
    st.markdown("""
    ### Practice technical interviews with AI feedback
    - Questions are automatically read aloud
    - Answer by typing or voice
    - Get instant AI scoring
    - Face monitoring throughout interview
    """)
    
    # Step 1: Face Verification
    st.divider()
    st.markdown("## Step 1: Face Verification")
    
    face_verified = simple_face_verification()
    
    if face_verified:
        st.divider()
        st.markdown("## Step 2: Interview Setup")
        
        # Load available roles
        try:
            questions_path = os.path.join("data", "questions.json")
            if os.path.exists(questions_path):
                with open(questions_path, 'r') as f:
                    data = json.load(f)
                    roles = list(data.keys())
            else:
                roles = ["Python Backend Developer"]
        except:
            roles = ["Python Backend Developer"]
        
        col1, col2 = st.columns(2)
        with col1:
            role = st.selectbox("Select Interview Role", roles)
        
        with col2:
            num_questions = st.slider("Number of Questions", 1, 10, 3)
        
        # Important note about face monitoring
        st.info("""
        ⚠️ **Note about face monitoring:**
        - Face monitoring will run continuously during the interview
        - Keep your face visible in the camera
        - The system will warn you if your face is not detected
        """)
        
        if st.button("🚀 Start Interview", type="primary", use_container_width=True):
            qs = get_interview_set(role)
            
            if qs:
                # Limit number of questions if needed
                if len(qs) > num_questions:
                    qs = random.sample(qs, num_questions)
                
                st.session_state.question_queue = qs
                st.session_state.results_log = []
                st.session_state.current_idx = 0
                st.session_state.app_state = 'interview'
                st.session_state.temp_ans = ""
                st.session_state.last_audio_played = -1
                st.session_state.face_warnings = 0
                st.session_state.last_face_detection = time.time()
                st.session_state.face_detection_log = []
                st.session_state.interview_started = True
                
                # Start continuous face monitoring
                if start_continuous_monitoring():
                    st.success("Face monitoring started!")
                else:
                    st.warning("Could not start camera, continuing without face monitoring")
                
                st.rerun()
            else:
                st.error("No questions found. Please check questions.json file.")

# =====================================================
# INTERVIEW PAGE
# =====================================================
elif st.session_state.app_state == 'interview':
    qs = st.session_state.question_queue
    idx = st.session_state.current_idx
    
    # Check if interview is complete or face warnings exceeded
    if idx >= len(qs) or st.session_state.face_warnings >= 3:
        stop_continuous_monitoring()
        st.session_state.app_state = 'report'
        st.rerun()
    
    current_q = qs[idx]
    
    # Two-column layout: Questions on left, Face monitor on right
    col_main, col_side = st.columns([3, 2])
    
    with col_side:
        # Show face monitoring panel
        st.markdown("### 🎥 Face Monitor")
        
        # Start monitoring if not already running
        if not face_monitor.running and st.session_state.interview_started:
            start_continuous_monitoring()
        
        # Show monitoring status
        monitoring_active = show_face_monitor_during_interview()
        
        # Show face warnings
        if st.session_state.face_warnings > 0:
            st.warning(f"⚠️ Face warnings: {st.session_state.face_warnings}")
        
        # Quick stats
        st.markdown("---")
        st.markdown("#### 📊 Interview Stats")
        
        # Progress
        progress = (idx + 1) / len(qs)
        st.progress(progress)
        st.caption(f"Question {idx + 1} of {len(qs)}")
        
        # Face detection rate
        if len(st.session_state.face_detection_log) > 0:
            recent_logs = st.session_state.face_detection_log[-20:]  # Last 20 checks
            if recent_logs:
                detection_rate = sum(1 for log in recent_logs if log["detected"]) / len(recent_logs) * 100
                st.metric("Face Visibility", f"{detection_rate:.0f}%")
        
        # Quick actions
        st.markdown("---")
        if st.button("🏁 Finish Interview", type="secondary"):
            stop_continuous_monitoring()
            st.session_state.app_state = 'report'
            st.rerun()
    
    with col_main:
        # ================= AUTOMATIC AUDIO PLAY =================
        if idx != st.session_state.last_audio_played:
            play_question_audio_automatically()
        
        # ================= QUESTION DISPLAY =================
        st.markdown(f"## Question {idx + 1}")
        st.markdown(f"### {current_q['text']}")
        
        # Show audio status
        if idx == st.session_state.last_audio_played:
            st.info("🔊 Question was read aloud")
        
        # Question type
        q_type = current_q.get('type', 'Technical')
        if q_type == 'Introduction':
            st.info("🤝 **Introduction Question**")
        else:
            st.warning("💻 **Technical Question**")
        
        # ================= ANSWER INPUT =================
        st.divider()
        st.markdown("### 📝 Your Answer")
        
        # Answer method
        answer_method = st.radio(
            "Answer using:",
            ["✍️ Type Answer", "🎤 Voice Answer"],
            horizontal=True,
            key=f"method_{idx}"
        )
        
        if answer_method == "✍️ Type Answer":
            manual_ans = st.text_area(
                "Type your answer here:",
                value=st.session_state.temp_ans,
                height=150,
                placeholder="Enter your detailed answer...",
                key=f"text_{idx}"
            )
            
            if manual_ans.strip():
                st.session_state.temp_ans = manual_ans
                
                col1, col2 = st.columns([3, 1])
                with col2:
                    if st.button("✅ Submit Answer", type="primary", use_container_width=True):
                        if manual_ans.strip():
                            process_answer(current_q, manual_ans.strip())
                        else:
                            st.warning("Please enter an answer before submitting.")
        
        else:  # Voice Answer
            st.info("Click below to record your answer")
            
            if st.button("🎤 Start Recording", use_container_width=True, key=f"record_{idx}"):
                with st.spinner("Listening... Speak now!"):
                    text = listen_audio()
                    if text:
                        st.session_state.temp_ans = text
                        st.success("✅ Answer recorded!")
                    else:
                        st.error("No speech detected. Please try again.")
            
            if st.session_state.temp_ans:
                st.text_area(
                    "Recorded Answer:",
                    st.session_state.temp_ans,
                    height=100,
                    disabled=True,
                    key=f"transcribed_{idx}"
                )
                
                if st.button("✅ Submit Recorded Answer", type="primary", key=f"submit_voice_{idx}"):
                    process_answer(current_q, st.session_state.temp_ans)
        
        # Show current answer if any
        if st.session_state.temp_ans:
            st.divider()
            st.markdown("**Your current answer:**")
            st.write(st.session_state.temp_ans)
        
        # ================= NAVIGATION =================
        st.divider()
        nav_col1, nav_col2, nav_col3 = st.columns(3)
        
        with nav_col1:
            if st.button("⏮️ Previous", disabled=idx == 0, use_container_width=True):
                st.session_state.current_idx = max(0, idx - 1)
                st.rerun()
        
        with nav_col2:
            if st.button("⏭️ Skip", use_container_width=True):
                st.session_state.results_log.append({
                    "question": current_q['text'],
                    "answer": "Skipped",
                    "score": 0,
                    "feedback": "Question skipped",
                    "face_detection_rate": 0
                })
                st.session_state.current_idx += 1
                st.rerun()
        
        with nav_col3:
            if st.button("🔄 Restart", type="secondary", use_container_width=True):
                stop_continuous_monitoring()
                st.session_state.app_state = 'setup'
                st.rerun()

# =====================================================
# REPORT PAGE
# =====================================================
elif st.session_state.app_state == 'report':
    # Make sure camera is stopped
    stop_continuous_monitoring()
    
    st.title("📊 Interview Report")
    
    results = st.session_state.results_log
    
    if not results:
        st.warning("No interview results found.")
        if st.button("Start New Interview"):
            st.session_state.clear()
            initialize_session_state()
            st.rerun()
    else:
        # Calculate scores
        scores = [r.get('score', 0) for r in results if isinstance(r.get('score'), (int, float))]
        if scores:
            avg_score = sum(scores) / len(scores)
        else:
            avg_score = 0
        
        # Calculate face monitoring metrics
        face_rates = [r.get('face_detection_rate', 0) for r in results]
        avg_face_rate = sum(face_rates) / len(face_rates) if face_rates else 0
        
        total_warnings = sum(r.get('face_warnings', 0) for r in results)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Overall Score", f"{avg_score:.1f}%")
        with col2:
            answered = len([r for r in results if r.get('answer', '').lower() != 'skipped'])
            st.metric("Answered", f"{answered}/{len(results)}")
        with col3:
            passed = sum(1 for r in results if r.get('score', 0) >= 70)
            st.metric("Passed", f"{passed}/{len(results)}")
        with col4:
            st.metric("Face Visibility", f"{avg_face_rate:.1f}%")
        
        # Face monitoring summary
        # Actions
        st.divider()
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("💾 Export Results (CSV)", use_container_width=True):
                export_data = []
                for i, res in enumerate(results):
                    export_data.append({
                        "question_number": i + 1,
                        "question": res['question'],
                        "answer": res['answer'],
                        "score": res['score'],
                        "feedback": res['feedback'],
                        "face_visibility": res.get('face_detection_rate', 0),
                        "face_warnings": res.get('face_warnings', 0)
                    })
                df = pd.DataFrame(export_data)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"interview_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        with col2:
            if st.button("📄 Export PDF Report", use_container_width=True):
                from fpdf import FPDF
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, txt="Interview Report", ln=True, align="C")
                pdf.ln(10)
                pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="L")
                pdf.ln(5)
                pdf.cell(200, 10, txt=f"Overall Score: {avg_score:.1f}%", ln=True, align="L")
                pdf.cell(200, 10, txt=f"Face Visibility: {avg_face_rate:.1f}%", ln=True, align="L")
                pdf.cell(200, 10, txt=f"Total Face Warnings: {total_warnings}", ln=True, align="L")
                pdf.ln(10)
                pdf.set_font("Arial", size=11)
                for i, res in enumerate(results):
                    pdf.multi_cell(0, 8, txt=f"Q{i+1}: {res['question']}", align="L")
                    pdf.multi_cell(0, 8, txt=f"Answer: {res['answer']}", align="L")
                    pdf.cell(0, 8, txt=f"Score: {res['score']}%", ln=True)
                    pdf.cell(0, 8, txt=f"Face Visibility: {res.get('face_detection_rate', 0):.0f}%", ln=True)
                    pdf.cell(0, 8, txt=f"Face Warnings: {res.get('face_warnings', 0)}", ln=True)
                    pdf.multi_cell(0, 8, txt=f"Feedback: {res['feedback']}", align="L")
                    pdf.ln(4)
                pdf_output = pdf.output(dest='S').encode('latin1')
                st.download_button(
                    label="Download PDF",
                    data=pdf_output,
                    file_name=f"interview_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

        with col3:
            if st.button("🔄 New Interview", type="primary", use_container_width=True, key="new_interview_col3"):
                st.session_state.clear()
                initialize_session_state()
                st.rerun()
                
                # Feedback
                feedback = res.get('feedback', '')
                if feedback:
                    if score >= 70:
                        st.success(f"**Feedback:** {feedback}")
                    elif score >= 50:
                        st.info(f"**Feedback:** {feedback}")
                    else:
                        st.warning(f"**Feedback:** {feedback}")
        
        # Actions
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Export Results", use_container_width=True):
                export_data = []
                for i, res in enumerate(results):
                    export_data.append({
                        "question_number": i + 1,
                        "question": res['question'],
                        "answer": res['answer'],
                        "score": res['score'],
                        "feedback": res['feedback'],
                        "face_visibility": res.get('face_detection_rate', 0),
                        "face_warnings": res.get('face_warnings', 0)
                    })
                
                df = pd.DataFrame(export_data)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"interview_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col2:
            if st.button("🔄 New Interview", type="primary", use_container_width=True, key="new_interview_bottom"):
                st.session_state.clear()
                initialize_session_state()
                st.rerun()