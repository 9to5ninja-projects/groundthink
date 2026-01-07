def detect_context_shift(hidden_states, threshold=0.8):
    """
    Detect when the topic changes dramatically and reset states.
    This prevents "carrying over" unrelated context.
    """
    if len(hidden_states) < 2:
        return False
    
    # Compute cosine similarity between recent states
    recent_states = hidden_states[-10:] if len(hidden_states) >= 10 else hidden_states
    
    similarities = []
    for i in range(1, len(recent_states)):
        cos_sim = F.cosine_similarity(
            recent_states[i].flatten(),
            recent_states[i-1].flatten(),
            dim=0
        )
        similarities.append(cos_sim.item())
    
    avg_similarity = np.mean(similarities)
    
    # If similarity drops dramatically, context has shifted
    if avg_similarity < threshold:
        print(f"⚠️ Context shift detected (similarity: {avg_similarity:.3f}), resetting states")
        return True
    
    return False

# In your model forward pass:
if detect_context_shift(self.hidden_state_history):
    self.reset_states()  # Start fresh for new topic