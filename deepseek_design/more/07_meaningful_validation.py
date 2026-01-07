def meaningful_validation(model, val_loader, device):
    """
    Don't just compute loss. These metrics tell you if your hybrid is actually working.
    """
    model.eval()
    metrics = {
        'perplexity': [],
        'state_coherence': [],  # Do states make sense over time?
        'context_retention': [],  # Can it remember things from earlier?
        'reasoning_accuracy': [],  # Simple logic tests
        'drift_score': []  # How much do states drift?
    }
    
    with torch.no_grad():
        for batch in val_loader:
            # Test 1: Simple pattern completion
            input_seq = batch['input_ids'].to(device)
            
            # Process first half
            half_len = input_seq.size(1) // 2
            first_half = input_seq[:, :half_len]
            _, states_first = model(first_half)
            
            # Process second half with states from first half
            second_half = input_seq[:, half_len:]
            logits_second, states_second = model(second_half, use_cache=True)
            
            # Compute context retention
            # If the model remembers the first half when processing second half
            retention_score = compute_retention(first_half, second_half, logits_second)
            metrics['context_retention'].append(retention_score)
            
            # Test 2: State coherence
            # Are the states evolving smoothly?
            coherence_score = compute_state_coherence(states_first, states_second)
            metrics['state_coherence'].append(coherence_score)
            
            # Test 3: Simple reasoning
            # Example: "If A then B. A. Therefore?" 
            reasoning_score = test_reasoning(model, device)
            metrics['reasoning_accuracy'].append(reasoning_score)
            
            # Test 4: Drift
            drift_score = compute_state_drift(states_first, states_second)
            metrics['drift_score'].append(drift_score)
            
            # Traditional perplexity
            loss = F.cross_entropy(logits_second.view(-1, logits_second.size(-1)),
                                  second_half.view(-1))
            perplexity = torch.exp(loss).item()
            metrics['perplexity'].append(perplexity)
    
    # Aggregate
    for key in metrics:
        if metrics[key]:
            metrics[key] = np.mean(metrics[key])
    
    model.train()
    return metrics

def compute_retention(first_half, second_half, logits):
    """
    Can the model use information from first half when processing second half?
    """
    # Simple check: Are tokens from first half influencing predictions in second half?
    # This is a simplified version - implement based on your specific needs
    return 0.5  # Placeholder

def compute_state_coherence(states1, states2):
    """Check if states evolve coherently"""
    total_coherence = 0
    count = 0
    
    for key in states1:
        if key in states2 and states1[key] is not None and states2[key] is not None:
            # Compute similarity between corresponding states
            sim = F.cosine_similarity(
                states1[key].flatten(),
                states2[key].flatten(),
                dim=0
            ).abs().item()
            total_coherence += sim
            count += 1
    
    return total_coherence / count if count > 0 else 0

def test_reasoning(model, device, num_tests=10):
    """Simple logical reasoning tests"""
    correct = 0
    
    for _ in range(num_tests):
        # Example: "The sky is blue. Grass is green. What color is the sky?"
        premise = "The sky is blue. Grass is green."
        question = "What color is the sky?"
        
        # Tokenize and process
        # This is simplified - you'd need actual tokenization
        input_ids = tokenize(premise + " " + question).to(device)
        
        with torch.no_grad():
            logits, _ = model(input_ids.unsqueeze(0))
        
        # Check if "blue" is a high-probability continuation
        # Simplified logic
        if "blue" in decode_top_predictions(logits[0, -1], top_k=3):
            correct += 1
    
    return correct / num_tests