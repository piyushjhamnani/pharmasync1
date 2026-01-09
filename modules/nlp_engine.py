import streamlit as st
try:
    from sentence_transformers import SentenceTransformer, util
    # Cache the model so it doesn't reload every time
    @st.cache_resource
    def load_bert():
        return SentenceTransformer('all-MiniLM-L6-v2')
    HAS_BERT = True
except:
    HAS_BERT = False

def compute_similarity(user_answer, model_answer):
    if not HAS_BERT:
        return 70, "BERT not installed. Using fallback score."
    
    # If user says nothing
    if not user_answer or len(user_answer.strip()) < 2: 
        return 0, "No answer provided."
    
    model = load_bert()
    # Math to compare meaning
    emb1 = model.encode(user_answer, convert_to_tensor=True)
    emb2 = model.encode(model_answer, convert_to_tensor=True)
    score = util.pytorch_cos_sim(emb1, emb2).item()
    
    final_score = int(max(0, score) * 100)
    
    if final_score > 80: feedback = "Excellent! Highly accurate."
    elif final_score > 50: feedback = "Good, but missed some details."
    else: feedback = "Your answer seems off-topic or incorrect."
    
    return final_score, feedback