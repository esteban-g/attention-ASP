# NOTES: In previous versions, it was penalized the speciazlied mask to make almost zero, now,  instead of a negative penalty on non-vocabulary tokens, I have updated the code to use a positive boost on the specialized vocabulary tokens. This produces the following effect (guessing)... 1. the function no longer applies a large negative penalty to tokens not in the vocabulary. 2. it now applies a positive BOOST_VALUE (10.0) to tokens that are in the specialized vocabulary (the key part here). 3, tokens outside the vocabulary receive a boost of 0.0, meaning their raw scaled dot-product scores are preserved. At the end, this strategy allows the raw attention scores (Q⋅K) to still influence the final weight for all tokens, but vocabulary tokens will receive a significant, targeted increase in relevance before the Softmax is applied. This should allow the underlying differences in the raw attention scores for each specialized head to surface, resulting in distinct mean attention scores. lets see

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer
import numpy as np

# ==============================================================================
# 1. Helper Function to Load Vocabularies
# ==============================================================================

def load_vocabulary(filepath):
    """Loads a vocabulary from a text file into a set for efficient lookup."""
    try:
        with open(filepath, 'r') as f:
            # Read lines, strip whitespace, and filter out empty lines
            terms = {line.strip().lower() for line in f if line.strip()}
        print(f"Successfully loaded {len(terms)} terms from {filepath}")
        return terms
    except FileNotFoundError:
        print(f"Error: Vocabulary file not found at {filepath}. Using empty set.")
        return set()

# ==============================================================================
# 2. The Modified Multi-Head Attention Module
# ==============================================================================

class SpecializedMultiHeadAttention(nn.Module):
    """
    A Multi-Head Attention module with a mix of standard and vocabulary-focused heads.
    """
    def __init__(self, d_model, n_heads, n_special_heads=4):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_special_heads = n_special_heads
        self.n_standard_heads = n_heads - n_special_heads
        self.d_k = d_model // n_heads

        # Linear projections for Q, K, V for all heads at once
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Final output projection
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, input_ids, tokenizer, special_vocabs):
        batch_size, seq_len, _ = x.shape

        # 1. Project inputs to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. Reshape and split into multiple heads
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 3. Separate standard and special heads
        q_std, q_spec = torch.split(q, [self.n_standard_heads, self.n_special_heads], dim=1)
        k_std, k_spec = torch.split(k, [self.n_standard_heads, self.n_special_heads], dim=1)
        v_std, v_spec = torch.split(v, [self.n_standard_heads, self.n_special_heads], dim=1)

        # --- Attention Calculation for Standard Heads ---
        scores_std = torch.matmul(q_std, k_std.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights_std = F.softmax(scores_std, dim=-1)
        output_std = torch.matmul(attn_weights_std, v_std)

        # --- Attention Calculation for Specialized Heads ---
        scores_spec_raw = torch.matmul(q_spec, k_spec.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Create the vocabulary mask (now a 'boost' mask)
        mask = self._create_vocab_mask(input_ids, tokenizer, special_vocabs, batch_size, seq_len)
        
        # Apply the boost to the attention scores before softmax
        scores_spec = scores_spec_raw + mask

        attn_weights_spec = F.softmax(scores_spec, dim=-1)
        output_spec = torch.matmul(attn_weights_spec, v_spec)

        # --- Compute Mean Attention Score for Specialized Heads ---
        head_mean_attention = attn_weights_spec.mean(dim=[-1, -2]).squeeze()

        # 4. Concatenate all head outputs and project
        output = torch.cat([output_std, output_spec], dim=1)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        final_output = self.out_proj(output)
        
        # Return final output, specialized attention weights, attention scores, and the new mean attention scores
        return final_output, attn_weights_spec, scores_std, scores_spec, head_mean_attention

    def _create_vocab_mask(self, input_ids, tokenizer, special_vocabs, batch_size, seq_len):
        """
        Creates a mask to guide attention towards specific vocabularies by *boosting* relevance.
        Instead of a negative penalty, we apply a positive value to vocabulary tokens.
        """
        # Get the actual tokens (strings) from their IDs
        tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]
        
        # Initialize mask tensor with a neutral value (0.0)
        vocab_mask = torch.zeros(batch_size, self.n_special_heads, 1, seq_len)
        
        # --- BOOST VALUE (Positive number) ---
        BOOST_VALUE = 10.0 

        for b in range(batch_size):
            for h in range(self.n_special_heads):
                current_vocab = special_vocabs[h]
                for s in range(seq_len):
                    # Clean the token (GPT-2 tokenizer adds 'Ġ' for spaces)
                    clean_token = tokens[b][s].replace('Ġ', '').lower()
                    
                    # Apply a positive boost if the token IS in the specialized vocabulary.
                    if clean_token in current_vocab:
                        vocab_mask[b, h, 0, s] = BOOST_VALUE
                        
        return vocab_mask.to(input_ids.device)

# ==============================================================================
# 3. Visualization Function (UPDATED)
# ==============================================================================

def plot_attention_heatmap(weights, tokens, vocab_names):
    """Plots a heatmap for every specialized attention head."""
    n_heads = weights.shape[0]
    # Calculate the number of rows and columns needed for the subplots
    cols = min(2, n_heads)
    rows = math.ceil(n_heads / cols)

    plt.figure(figsize=(6 * cols, 5 * rows)) # Adjust figure size dynamically

    # Ensure the tokens are clean and replace any unexpected Unicode characters
    clean_tokens = [token.replace('Ġ', '').replace('Ĝ', 'G').replace('Â', '') for token in tokens]

    for i in range(n_heads):
        plt.subplot(rows, cols, i + 1)
        
        # Use an appropriate colormap for attention weights
        sns.heatmap(
            weights[i],
            xticklabels=clean_tokens,
            yticklabels=clean_tokens,
            annot=False,
            cmap='viridis',
            cbar=True, # Show color bar for context
            vmin=0.0,  # Attention weights are [0, 1]
            vmax=1.0
        )
        plt.title(f'Specialized Head: {vocab_names[i].upper()}', fontsize=10)
        plt.xlabel('Key Tokens', fontsize=8)
        plt.ylabel('Query Tokens', fontsize=8)
        plt.xticks(rotation=45, ha='right', fontsize=6)
        plt.yticks(rotation=0, fontsize=6)

    plt.tight_layout()
    plt.show()

# ==============================================================================
# 4. Main Execution Block
# ==============================================================================

if __name__ == "__main__":
    # --- Configuration ---
    D_MODEL = 512
    N_HEADS = 8
    N_SPECIAL_HEADS = 4
    
    # --- Load Tokenizer ---
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    vocab_size = tokenizer.vocab_size

    # --- Load Specialized Vocabularies ---
    vocab_files = {
        'time': 'time.dic',
        'place': 'place.dic',
        'people': 'people.dic',
        'dedication': 'dedication.dic',
        'investment': 'investment.dic'
    }
    required_vocabs = {k: v for i, (k, v) in enumerate(vocab_files.items()) if i < N_SPECIAL_HEADS}
    vocab_names = list(required_vocabs.keys())
    
    # Load vocabularies from files (assuming they are present)
    special_vocabs = [load_vocabulary(f) for f in required_vocabs.values()]


    # --- Sample Input ---
    sample_text = ("When I go to football practice, I usually feel a bit out of place."
                    "Even though I'm on the field with the other kids, I often find myself distracted." 
                    "Sometimes I just sit on the bench and scroll through my phone instead of joining the game." 
                    "I try to play, but it's hard to focus, and I don't feel very good when I'm out there."
                    "It gets even worse when Leo arrives-he's really fast and loud, and when he's around, I feel like I don't belong at all."
                    "I know the coaches want me to participate more, but it's not easy."
                    "I feel like I'm just there, not really part of the team.") 
    print("\n--- Input ---")
    print(f"Sample Text: {sample_text}")

    # --- Prepare Model and Inputs ---
    inputs = tokenizer(sample_text, return_tensors='pt', padding=True)
    input_ids = inputs['input_ids']
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    embedding = nn.Embedding(vocab_size, D_MODEL)
    input_embeddings = embedding(input_ids)

    special_attention_block = SpecializedMultiHeadAttention(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_special_heads=N_SPECIAL_HEADS
    )

    # --- Run the Model and capture the output ---
    output, special_attn_weights, standard_scores, specialized_scores, mean_attention_scores = special_attention_block(
        x=input_embeddings,
        input_ids=input_ids,
        tokenizer=tokenizer,
        special_vocabs=special_vocabs
    )
    
    print("\n--- Output ---")
    
    # --- Print Mean Attention Scores ---
    print("\n--- Mean Attention Score per Specialized Head (Scaled [0, 1]) ---")
    
    scores = mean_attention_scores.detach().cpu().numpy()
    
    for name, score in zip(vocab_names, scores):
        print(f"**{name.upper():<10}** Score: {score:.6f}")
        
    print("-" * 50)
    print(f"Standard Heads Scaled Dot-Product Scores Shape (B, H_std, S, S): {standard_scores.shape}")
    print(f"Specialized Heads Boosted Scaled Dot-Product Scores Shape (B, H_spec, S, S): {specialized_scores.shape}")

    # --- Print Slices of Scores (for context) ---
    if N_HEADS - N_SPECIAL_HEADS > 0:
        print("\n--- Scaled Dot-Product Scores (Standard Heads, First Head, Top-Left 3x3 Slice) ---")
        print(standard_scores[0, 0, :3, :3].detach().cpu().numpy())

    if N_SPECIAL_HEADS > 0:
        print("\n--- Boosted Scaled Dot-Product Scores (Specialized Heads, First Head, Top-Left 3x3 Slice) ---")
        print(specialized_scores[0, 0, :3, :3].detach().cpu().numpy())

    # --- Visualize the Attention (UPDATED) ---
    # We take the weights from the first (and only) item in the batch
    weights_to_plot = special_attn_weights[0].detach().cpu().numpy()
    plot_attention_heatmap(weights_to_plot, tokens, vocab_names)