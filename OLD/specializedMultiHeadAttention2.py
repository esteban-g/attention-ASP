import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer

# Helper Function to Load Vocabularies
def load_vocabulary(filepath):
    try:
        with open(filepath, 'r') as f:
            terms = {line.strip().lower() for line in f if line.strip()}
        print(f"Successfully loaded {len(terms)} terms from {filepath}")
        return terms
    except FileNotFoundError:
        print(f"Error: Vocabulary file not found at {filepath}")
        return set()

# Modified Multi-Head Attention Module
class SpecializedMultiHeadAttention(nn.Module):
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

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights

    def forward(self, x, input_ids, tokenizer, special_vocabs):
        batch_size, seq_len, _ = x.shape

        # Project input to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Split into standard and specialized heads
        q_std, q_spec = q.split([self.n_standard_heads, self.n_special_heads], dim=1)
        k_std, k_spec = k.split([self.n_standard_heads, self.n_special_heads], dim=1)
        v_std, v_spec = v.split([self.n_standard_heads, self.n_special_heads], dim=1)

        # Compute scaled dot product attention for standard heads
        std_output, std_attn_weights = self.scaled_dot_product_attention(q_std, k_std, v_std)

        # Compute scaled dot product attention for specialized heads
        spec_output, spec_attn_weights = self.scaled_dot_product_attention(q_spec, k_spec, v_spec)

        # Concatenate outputs
        output = torch.cat([std_output, spec_output], dim=1)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(output)

        return output, std_attn_weights, spec_attn_weights

# Example Usage
D_MODEL = 512
N_HEADS = 8
N_SPECIAL_HEADS = 4
vocab_size = 10000

# Load Specialized Vocabularies
vocab_files = {
    'Tech': 'tech_vocab.txt',
    'Science': 'science_vocab.txt',
    'Finance': 'finance_vocab.txt',
    'Art': 'art_vocab.txt',
}
vocab_names = list(vocab_files.keys())
special_vocabs = [load_vocabulary(f) for f in vocab_files.values()]

# Sample Input
sample_text = (
    "The transformer architecture uses python, while the galaxy contains a neuron. "
    "He made an investment in the crypto market. That painting is a great sculpture."
)

# Prepare Model and Inputs
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # Set the padding token

inputs = tokenizer(sample_text, return_tensors='pt', padding=True)
input_ids = inputs['input_ids']
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

# Create a simple embedding layer
embedding = nn.Embedding(vocab_size, D_MODEL)
input_embeddings = embedding(input_ids)

# Instantiate our custom attention module
special_attention_block = SpecializedMultiHeadAttention(
    d_model=D_MODEL,
    n_heads=N_HEADS,
    n_special_heads=N_SPECIAL_HEADS
)

# Run the Model
output, std_attn_weights, spec_attn_weights = special_attention_block(
    x=input_embeddings,
    input_ids=input_ids,
    tokenizer=tokenizer,
    special_vocabs=special_vocabs
)

print("\n--- Output ---")
print(f"Model Output Shape: {output.shape}")
print(f"Standard Attention Weights Shape: {std_attn_weights.shape}")
print(f"Specialized Attention Weights Shape: {spec_attn_weights.shape}")
