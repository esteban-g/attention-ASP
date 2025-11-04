# NOTES: In previous versions, there was a mistake about the overall score, which was identical for every head, which was a mistake. Now, to get a scaled number between $[0, 1]$ that actually reflects the degree of focus and breaks the $1/S$ uniformity, it is necessary  to measure the concentration of attention on the vocabulary tokens. The appropriate metric is the average attention weight given to tokens that are in the vocabulary. For a specialized head $h$ and a given input sentence, the best single score is: $$\text{Concentration Score}_h = \frac{\sum_{i=1}^{S} \sum_{j \in \text{Vocab}_h} \mathbf{A}_{h, i, j}}{S}$$ This metric sums the weights only for the vocabulary columns, and then averages that sum over the $S$ Query tokens. A high score means a large portion of the model's attention budget was spent on the specialized vocabulary.


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

        # Create the vocabulary mask (the 'boost' mask)
        mask = self._create_vocab_mask(input_ids, tokenizer, special_vocabs, batch_size, seq_len)
        
        # Apply the boost to the attention scores before softmax
        scores_spec = scores_spec_raw + mask

        attn_weights_spec = F.softmax(scores_spec, dim=-1)
        output_spec = torch.matmul(attn_weights_spec, v_spec)

        # --- NEW: Compute Concentration Score for Specialized Heads ---
        concentration_mask = self._create_binary_concentration_mask(input_ids, tokenizer, special_vocabs, batch_size, seq_len)
        
        # 1. Filter attention weights: weights_focused only contains values for vocabulary tokens
        weights_focused = attn_weights_spec * concentration_mask
        
        # 2. Sum weights over the Key dimension (columns: dim=-1). This gives the total attention 
        #    paid to vocabulary tokens for *each* Query token (row). Result shape: (B, H_spec, S)
        sum_per_query = weights_focused.sum(dim=-1)
        
        # 3. Average the sums over the Query dimension (rows: dim=-1). This is the final Concentration Score.
        head_concentration_score = sum_per_query.mean(dim=-1).squeeze()

        # 4. Concatenate all head outputs and project
        output = torch.cat([output_std, output_spec], dim=1)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        final_output = self.out_proj(output)
        
        # Return final output, specialized attention weights, attention scores, and the new concentration scores
        return final_output, attn_weights_spec, scores_std, scores_spec, head_concentration_score

    def _create_vocab_mask(self, input_ids, tokenizer, special_vocabs, batch_size, seq_len):
        """
        Creates the boost mask used for the attention calculation.
        """
        tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]
        vocab_mask = torch.zeros(batch_size, self.n_special_heads, 1, seq_len)
        BOOST_VALUE = 10.0 

        for b in range(batch_size):
            for h in range(self.n_special_heads):
                current_vocab = special_vocabs[h]
                for s in range(seq_len):
                    clean_token = tokens[b][s].replace('Ġ', '').lower()
                    if clean_token in current_vocab:
                        vocab_mask[b, h, 0, s] = BOOST_VALUE
                        
        return vocab_mask.to(input_ids.device)

    def _create_binary_concentration_mask(self, input_ids, tokenizer, special_vocabs, batch_size, seq_len):
        """
        Creates a binary mask (1.0 for vocabulary, 0.0 otherwise) for the Concentration Score calculation.
        Shape: (B, n_special_heads, 1, S)
        """
        tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]
        # Mask tensor initialized to 0.0
        concentration_mask = torch.zeros(batch_size, self.n_special_heads, 1, seq_len)
        
        for b in range(batch_size):
            for h in range(self.n_special_heads):
                current_vocab = special_vocabs[h]
                for s in range(seq_len):
                    clean_token = tokens[b][s].replace('Ġ', '').lower()
                    
                    # Set mask to 1.0 if the token is in the specialized vocabulary.
                    if clean_token in current_vocab:
                        concentration_mask[b, h, 0, s] = 1.0
                        
        # The mask needs to be broadcastable to the S x S attention matrix.
        # It's currently (B, H_spec, 1, S) which works by broadcasting the Query (row) dim.
        # We need to explicitly repeat it across the Query dimension to create an S x S mask:
        # Result shape: (B, H_spec, S, S)
        return concentration_mask.repeat(1, 1, seq_len, 1).to(input_ids.device)


# ==============================================================================
# 3. Visualization Function 
# ==============================================================================

def plot_attention_heatmap(weights, tokens, vocab_names):
    """Plots a heatmap for every specialized attention head."""
    n_heads = weights.shape[0]
    cols = min(2, n_heads)
    rows = math.ceil(n_heads / cols)

    plt.figure(figsize=(6 * cols, 5 * rows)) 

    clean_tokens = [t.replace('Ġ', '').replace('Ĝ', 'G').replace('Â', '') for t in tokens]

    for i in range(n_heads):
        plt.subplot(rows, cols, i + 1)
        
        sns.heatmap(
            weights[i],
            xticklabels=clean_tokens,
            yticklabels=clean_tokens,
            annot=False,
            cmap='viridis',
            cbar=True,
            vmin=0.0,
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
        'activity': 'dedication.dic',
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
    output, special_attn_weights, standard_scores, specialized_scores, concentration_scores = special_attention_block(
        x=input_embeddings,
        input_ids=input_ids,
        tokenizer=tokenizer,
        special_vocabs=special_vocabs
    )
    
    print("\n--- Output ---")
    
    # --- Print Concentration Scores ---
    print("\n--- Concentration Score per Specialized Head (Scaled [0, 1]) ---")
    
    scores = concentration_scores.detach().cpu().numpy()
    
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

    # --- Visualize the Attention ---
    weights_to_plot = special_attn_weights[0].detach().cpu().numpy()
    plot_attention_heatmap(weights_to_plot, tokens, vocab_names)