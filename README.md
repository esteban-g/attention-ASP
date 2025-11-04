# attention-ASP: A Neuro-Symbolic Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Built with PyTorch](https://img.shields.io/badge/Built%20with-PyTorch-FF5722?logo=pytorch)](https://pytorch.org/)
[![Built with Clingo](https://img.shields.io/badge/Built%20with-Clingo-blueviolet)](https://potassco.org/clingo/)

A research prototype for bridging sub-symbolic neural attention with symbolic logical inference. This framework integrates a **Specialized Multi-Head Attention** mechanism with **Answer Set Programming (ASP)** to perform logical reasoning based on a model's attention patterns in a text.

This project is the basis for the methodology section of our article submitted to MIE2026 conference (under review).

## 🧠 Core Concept

This project bridges the gap between the continuous, probabilistic world of Transformers and the discrete, logical world of symbolic AI.

It implements a modified **Multi-Head Attention (MHA)** block where specific heads are "specialized" to focus on predefined vocabularies (e.g., words related to 'time', 'place', 'people'), these masks were captured and modified by ASP. Part of the novelty of **attention-ASP** is the **Attention Concentration Score ($\mathbf{C}_h$)**, a scalar value that quantifies how much attention each specialized head pays to its respective vocabulary in a given text. This score is then used as continuous evidence to **ground discrete logical atoms** in an Answer Set Programming (ASP) program, which is solved using **Clingo**.

### How It Works

1.  **Input Text**: A raw text (e.S., a client's transcript) is provided.
2.  **Specialized Attention**: The text is passed through a modified Transformer block. Specialized heads are "boosted" to focus on their respective vocabularies (e.g., `time.dic`, `place.dic`).
3.  **Concentration Score**: The model calculates a $\mathbf{C}_h$ score for each head (e.g., `time_score = 0.9992`, `place_score = 0.9967`).
4.  **Grounding**: These scores are compared against a threshold (e.g., $\tau=0.1$) to create logical facts (e.g., `high_time.`, `in_place.`).
5.  **ASP Inference**: The Clingo solver runs an ASP program (e.g., `hook_involv :- high_time, in_place, ...`) using these grounded facts.
6.  **Logical Output**: The system outputs a stable model, providing a discrete, logical, and human-interpretable conclusion (e.g., `hook_involv` is `TRUE`).

## 🛠️ Installation and Setup

To get started, clone the repository and create the Conda environment using the provided `environment.yml` file.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/attention-ASP.git](https://github.com/YOUR_USERNAME/attention-ASP.git)
    cd attention-ASP
    ```

2.  **Create and activate the Conda environment:**
    ```bash
    # Create the environment from the file
    conda env create -f environment.yml
    
    # Activate the new environment
    conda activate special_attn_env
    ```

## 🚀 Usage

Before running the main script, you must provide the specialized vocabularies.

1.  **Create Vocabulary Files:**
    Create `.dic` or `.txt` files in the root directory. Each file should contain a list of terms (one per line) for a specific concept.

    *Example: `time.dic`*
    ```
    when
    usually
    sometimes
    ```

    *Example: `place.dic`*
    ```
    field
    bench
    out
    there
    ```

2.  **Run the Experiment:**
    Execute the main Python script. The script will load the text, compute the attention scores, run the Clingo solver, and output the logical inference results.

    ```bash
    python specializedMultiHeadAttention6B.py
    ```

### Example Output
    ```
        --- Input ---
    Sample Text: When I go to football practice, I usually feel a bit out of place.Even though I'm on the field with the other kids, I often find myself distracted.Sometimes I just sit on the bench and scroll through my phone instead of joining the game.I try to play, but it's hard to focus, and I don't feel very good when I'm out there.It gets even worse when Leo arrives-he's really fast and loud, and when he's around, I feel like I don't belong at all.I know the coaches want me to participate more, but it's not easy.I feel like I'm just there, not really part of the team.

    --- Output ---

    --- Concentration Score per Specialized Head (Scaled [0, 1]) ---
    **TIME      ** Score: 0.999158
    **PLACE     ** Score: 0.996610
    **PEOPLE    ** Score: 0.992941
    **ACTIVITY  ** Score: 0.996844
    --------------------------------------------------
    Standard Heads Scaled Dot-Product Scores Shape (B, H_std, S, S): torch.Size([1, 4, 139, 139])
    Specialized Heads Boosted Scaled Dot-Product Scores Shape (B, H_spec, S, S): torch.Size([1, 4, 139, 139])

    --- Scaled Dot-Product Scores (Standard Heads, First Head, Top-Left 3x3 Slice) ---
    [[-0.6269649   0.10452973  0.03445041]
    [ 0.19512819 -0.10101509  0.20195116]
    [-0.18912654  0.6461076  -0.5914499 ]]

    --- Boosted Scaled Dot-Product Scores (Specialized Heads, First Head, Top-Left 3x3 Slice) ---
    [[ 9.622966    0.1823554  -0.80838925]
    [10.092496   -0.50624293  0.2635796 ]
    [10.026835   -0.24686176 -0.51120406]]

    ```

    ![alt text](https://github.com/esteban-g/attention-ASP/tree/main/blob/main/img/Figure_1_MIE2026.png?raw=true)
