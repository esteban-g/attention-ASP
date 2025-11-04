import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer

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
        print(f"Error: Vocabulary file not found at {filepath}")
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
        q = self.q_proj(x) # (B, S, D)
        k = self.k_proj(x) # (B, S, D)
        v = self.v_proj(x) # (B, S, D)

        # 2. Reshape and split into multiple heads
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2) # (B, H, S, d_k)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2) # (B, H, S, d_k)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2) # (B, H, S, d_k)

        # 3. Separate standard and special heads
        q_std, q_spec = torch.split(q, [self.n_standard_heads, self.n_special_heads], dim=1)
        k_std, k_spec = torch.split(k, [self.n_standard_heads, self.n_special_heads], dim=1)
        v_std, v_spec = torch.split(v, [self.n_standard_heads, self.n_special_heads], dim=1)

        # --- Attention Calculation for Standard Heads ---
        scores_std = torch.matmul(q_std, k_std.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights_std = F.softmax(scores_std, dim=-1)
        output_std = torch.matmul(attn_weights_std, v_std)

        # --- Attention Calculation for Specialized Heads ---
        scores_spec = torch.matmul(q_spec, k_spec.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Create the vocabulary mask
        # The mask will heavily penalize tokens *not* in the desired vocabulary
        mask = self._create_vocab_mask(input_ids, tokenizer, special_vocabs, batch_size, seq_len)
        
        # Apply the mask to the attention scores before softmax
        # Mask shape: (B, n_special_heads, 1, S), scores_spec shape: (B, n_special_heads, S, S)
        # Broadcasting will apply the mask across the query dimension (dim=-2)
        scores_spec = scores_spec + mask

        attn_weights_spec = F.softmax(scores_spec, dim=-1)
        output_spec = torch.matmul(attn_weights_spec, v_spec)

        # 4. Concatenate all head outputs and project
        output = torch.cat([output_std, output_spec], dim=1)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        final_output = self.out_proj(output)
        
        # Return final output and the attention weights of special heads for visualization
        return final_output, attn_weights_spec

    def _create_vocab_mask(self, input_ids, tokenizer, special_vocabs, batch_size, seq_len):
        """Creates a mask to guide attention towards specific vocabularies."""
        # Get the actual tokens (strings) from their IDs
        tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]
        
        # Initialize mask tensor
        # Shape: (B, n_special_heads, 1, S)
        vocab_mask = torch.zeros(batch_size, self.n_special_heads, 1, seq_len)

        for b in range(batch_size):
            for h in range(self.n_special_heads):
                current_vocab = special_vocabs[h]
                for s in range(seq_len):
                    # Clean the token (GPT-2 tokenizer adds 'Ġ' for spaces)
                    clean_token = tokens[b][s].replace('Ġ', '').lower()
                    if clean_token not in current_vocab:
                        # Apply a large negative value to mask this token out
                        vocab_mask[b, h, 0, s] = -1e9
                        
        return vocab_mask.to(input_ids.device)

# ==============================================================================
# 3. Visualization Function
# ==============================================================================

def plot_attention_heatmapX(attention_weights, tokens, vocab_names):
    """Visualizes the attention weights for the specialized heads."""
    assert len(attention_weights) == len(vocab_names), "Mismatch between weights and names"
    
    num_heads = len(attention_weights)
    fig, axes = plt.subplots(1, num_heads, figsize=(20, 5), dpi=100)
    
    if num_heads == 1:
        axes = [axes] # Make it iterable for a single head
        
    for i, ax in enumerate(axes):
        sns.heatmap(attention_weights[i], xticklabels=tokens, yticklabels=tokens,
                    annot=True, fmt=".2f", cmap="viridis", ax=ax, cbar=False, annot_kws={"size": 4})        
        ax.set_title(f"Attention Head for '{vocab_names[i]}' Vocab")
        ax.tick_params(axis='x', rotation=90)
        ax.tick_params(axis='y', rotation=0)

    fig.tight_layout()
    plt.show()


def plot_attention_heatmap(weights, tokens, vocab_names):
    plt.figure(figsize=(12, 10))

    # Ensure the tokens are clean and replace any unexpected Unicode characters
    clean_tokens = [token.replace('Ĝ', 'G').replace('Â', '') for token in tokens]

    for i, head_weights in enumerate(weights):
        plt.subplot(2, 2, i + 1)
        sns.heatmap(
            head_weights,
            xticklabels=clean_tokens,
            yticklabels=clean_tokens,
            annot=True,
            annot_kws={"size": 6},  # Smaller annotation font size
            cmap='viridis'
        )
        plt.title(f'Attention Head: {vocab_names[i]}')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

    plt.tight_layout()
    plt.show()

# ==============================================================================
# 4. Main Execution Block
# ==============================================================================

if __name__ == "__main__":
    # --- Configuration ---
    D_MODEL = 512       # Embedding dimension
    N_HEADS = 8         # Total number of attention heads
    N_SPECIAL_HEADS = 4 # Number of specialized heads (must be <= N_HEADS)
    
    # --- Load Tokenizer ---
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # Add a padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    vocab_size = tokenizer.vocab_size

    # --- Load Specialized Vocabularies ---
    # vocab_files = {
    #     'Tech': 'tech_vocab.txt',
    #     'Science': 'science_vocab.txt',
    #     'Finance': 'finance_vocab.txt',
    #     'Art': 'art_vocab.txt',
    # }
    vocab_files = {
        'time': 'time.dic',
        'place': 'place.dic',
        'people': 'people.dic',
        'dedication': 'dedication.dic',
        'investment': 'investment.dic'
    }
    vocab_names = list(vocab_files.keys())
    special_vocabs = [load_vocabulary(f) for f in vocab_files.values()]

    # --- Sample Input ---
    # sample_text = ("The transformer architecture uses python, while the galaxy contains a neuron. "
    #                "He made an investment in the crypto market. That painting is a great sculpture.")
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
    # 1. Tokenize the input text
    inputs = tokenizer(sample_text, return_tensors='pt', padding=True)
    input_ids = inputs['input_ids']
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    print(f"Tokens: {tokens}")
    
    # 2. Create a simple embedding layer
    embedding = nn.Embedding(vocab_size, D_MODEL)
    input_embeddings = embedding(input_ids)

    # 3. Instantiate our custom attention module
    special_attention_block = SpecializedMultiHeadAttention(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_special_heads=N_SPECIAL_HEADS
    )

    # --- Run the Model ---
    # The 'special_vocabs' list must be in the same order as the special heads
    output, special_attn_weights = special_attention_block(
        x=input_embeddings,
        input_ids=input_ids,
        tokenizer=tokenizer,
        special_vocabs=special_vocabs
    )
    
    print("\n--- Output ---")
    print(f"Model Output Shape: {output.shape}")
    print(f"Special Attention Weights Shape: {special_attn_weights.shape}") # (B, H_spec, S, S)
    
    # --- Visualize the Attention ---
    # We take the weights from the first (and only) item in the batch
    weights_to_plot = special_attn_weights[0].detach().cpu().numpy()
    plot_attention_heatmap(weights_to_plot, tokens, vocab_names)