from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def compute_response_probabilities_batch(target_model, tokenized_prompts, response_candidates):
    """
    Compute probabilities for each response candidate without explicit loops.

    Args:
        target_model: Hugging Face transformer model (AutoModelForCausalLM or similar).
        tokenized_prompts: Tensor of shape (B, max_seq_len), tokenized input prompts.
        response_candidates: List of K tokenized candidate responses (each tensor).
    
    Returns:
        Tensor of shape (B, K) representing probabilities for each response candidate.
    """
    prompt_attention_mask = tokenized_prompts.attention_mask # Shape: (B, max_seq_len)
    tokenized_prompts = tokenized_prompts.input_ids # Shape: (B, max_seq_len)

    response_candidates_attention_mask = response_candidates.attention_mask # Shape: (K, max_candidate_len)
    response_candidates = response_candidates.input_ids
    device = target_model.device
    # Convert response candidates to a single tensor with padding
    max_candidate_len = response_candidates.shape[1]

    batch_size, max_seq_len = tokenized_prompts.size()
    num_candidates = len(response_candidates)
    
    # Expand tokenized prompts to match response candidates
    tokenized_prompts_expanded = tokenized_prompts.unsqueeze(1).expand(batch_size, num_candidates, max_seq_len)
    response_candidates_expanded = response_candidates.unsqueeze(0).expand(batch_size, num_candidates, max_candidate_len)

    # Concatenate prompts and responses
    concatenated_inputs = torch.cat([tokenized_prompts_expanded, response_candidates_expanded], dim=-1)
    concatenated_inputs = concatenated_inputs.view(-1, concatenated_inputs.size(-1))  # Flatten to (B*K, sequence_len)

    attention_mask = torch.cat([prompt_attention_mask.unsqueeze(1).expand(batch_size, num_candidates, max_seq_len), response_candidates_attention_mask.unsqueeze(0).expand(batch_size, num_candidates, max_candidate_len)], dim=-1)
    attention_mask = attention_mask.view(-1, attention_mask.size(-1)) # Flatten to (B*K, sequence_len)
    with torch.no_grad():
        outputs = target_model(input_ids=concatenated_inputs.to(device), attention_mask=attention_mask.to(device), labels=concatenated_inputs.to(device))
        losses = outputs.loss  # Shape: scalar
    print(outputs.logits.shape)
    print(concatenated_inputs.shape)
    logits = outputs.logits #[6, 10, 128256])
    # fill attention_mask = 0 with
    per_token_logit = logits.gather(2, concatenated_inputs.unsqueeze(-1)).squeeze(-1) # Shape: (B*K, sequence_len)
    per_token_logit_after_attention_mask = per_token_logit * attention_mask
    print(per_token_logit.shape)
    print(per_token_logit)
    # Compute probabilities
    probabilities = torch.nn.functional.softmax(per_token_logit, dim=-1)
    return probabilities


def compute_response_probabilities_old(target_model, tokenized_prompts, response_candidates):
    """
    Compute probabilities for each response candidate.

    Args:
        target_model: Hugging Face transformer model (AutoModelForCausalLM or similar).
        tokenized_prompts: Tensor of shape (B, max_seq_len), tokenized input prompts.
        response_candidates: List of K tokenized candidate responses.

    Returns:
        Tensor of shape (B, K) representing probabilities for each response candidate.
    """
    device =target_model.device

    batch_size, max_seq_len = tokenized_prompts.input_ids.size()
    num_candidates = len(response_candidates)
    probabilities = torch.zeros(batch_size, num_candidates).to(device)

    prompt_attention_mask = tokenized_prompts.attention_mask
    response_candidates_attention_mask = response_candidates.input_ids

    with torch.no_grad():
        for b in range(batch_size):
            for k, response in enumerate(response_candidates):
                # Concatenate the prompt and response
            
                input_ids = torch.cat([tokenized_prompts.input_ids[b], response])
                input_ids = input_ids.unsqueeze(0).to(device)  # Add batch dimension
                attention_mask = torch.cat([prompt_attention_mask[b], response_candidates_attention_mask[k]]).unsqueeze(0).to(device) 
                # Get the model outputs
                outputs = target_model(input_ids, labels=input_ids, attention_mask=attention_mask)
                loss = outputs.loss  # Cross-entropy loss
                probabilities[b, k] = torch.exp(-loss)  # Convert loss to probability

    return probabilities


def compute_response_probabilities(target_model, tokenized_prompts, response_candidates):
    """
    Compute probabilities for each response candidate, accounting for attention masks.

    Args:
        target_model: Hugging Face transformer model (AutoModelForCausalLM or similar).
        tokenized_prompts: Dictionary with 'input_ids' (N, seq_len) and 'attention_mask' (N, seq_len).
        response_candidates: Dictionary with 'input_ids' (K, response_len) and 'attention_mask' (K, response_len).
    
    Returns:
        Tensor of shape (N, K) representing probabilities for each response candidate.
    """
    device = target_model.device

    # Extract prompts and candidates
    prompt_input_ids = tokenized_prompts["input_ids"].to(device)  # Shape: (N, seq_len)
    prompt_attention_mask = tokenized_prompts["attention_mask"].to(device)  # Shape: (N, seq_len)
    
    candidate_input_ids = response_candidates["input_ids"].to(device)  # Shape: (K, response_len)
    candidate_attention_mask = response_candidates["attention_mask"].to(device)  # Shape: (K, response_len)

    num_prompts, prompt_len = prompt_input_ids.size()
    num_candidates, candidate_len = candidate_input_ids.size()

    # Expand prompts to match candidates
    expanded_prompt_input_ids = prompt_input_ids.unsqueeze(1).expand(num_prompts, num_candidates, prompt_len)
    expanded_prompt_attention_mask = prompt_attention_mask.unsqueeze(1).expand(num_prompts, num_candidates, prompt_len)

    # Expand candidates to match prompts
    expanded_candidate_input_ids = candidate_input_ids.unsqueeze(0).expand(num_prompts, num_candidates, candidate_len)
    expanded_candidate_attention_mask = candidate_attention_mask.unsqueeze(0).expand(num_prompts, num_candidates, candidate_len)

    # Concatenate prompts and candidates
    concatenated_input_ids = torch.cat([expanded_prompt_input_ids, expanded_candidate_input_ids], dim=-1)
    concatenated_attention_mask = torch.cat([expanded_prompt_attention_mask, expanded_candidate_attention_mask], dim=-1)

    # Flatten for batch processing
    concatenated_input_ids = concatenated_input_ids.view(-1, concatenated_input_ids.size(-1))  # Shape: (N*K, seq_len + response_len)
    concatenated_attention_mask = concatenated_attention_mask.view(-1, concatenated_attention_mask.size(-1))  # Shape: (N*K, seq_len + response_len)

    # Prepare labels (same as input_ids, but pad tokens are ignored)
    labels = concatenated_input_ids.clone()
    labels[concatenated_attention_mask == 0] = -100  # Ignore padding tokens in loss calculation

    with torch.no_grad():
        outputs = target_model(input_ids=concatenated_input_ids, attention_mask=concatenated_attention_mask)
        logits = outputs.logits  # Shape: (N*K, seq_len + response_len, vocab_size)

    # Compute element-wise cross-entropy loss
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    token_losses = token_losses.view(shift_labels.size())  # Shape: (N*K, seq_len + response_len - 1)

    sequence_losses = token_losses[:,prompt_len-1:].sum(dim=-1)   # Sum losses for response tokens only
    sequence_losses = sequence_losses.view(num_prompts, num_candidates)  # Reshape to (N, K)

    probabilities = torch.exp(-sequence_losses)  # Convert loss to probabilities

    return probabilities


model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tokenizer.pad_token = tokenizer.eos_token
# Inputs
prompts = ["What is the capital of UK?", "Who wrote '1984'?"]
responses = ["Paris", "George Orwell", "London"]

# Tokenize inputs
tokenized_prompts = tokenizer(text=prompts, 
                            truncation=True,
                            padding='longest',
                            add_special_tokens=True,
                            return_tensors='pt',
                            return_attention_mask=True)


response_candidates = tokenizer(responses, truncation=True, padding='longest', return_tensors='pt', return_attention_mask=True,add_special_tokens=False)
print(response_candidates, tokenized_prompts)
# Compute probabilities
probs = compute_response_probabilities(model, tokenized_prompts, response_candidates)
print(probs) # shape (N, K)
# select the best response for each prompt
best_responses = [responses[i] for i in probs.argmax(dim=1)]
for prompt, response in zip(prompts, best_responses):
    print(f"Prompt: {prompt}\nBest response: {response}\n")

