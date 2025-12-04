# Dialectical-TTS: Local Reasoning Engine

**A local implementation of Test-Time Compute (Reasoning) for Apple Silicon.**

This is not a chatbot. It is a structural reasoning engine that uses **Tri-State Dialectics** (Thesis, Antithesis, Synthesis) to overcome the probabilistic biases of Large Language Models.

Designed to run on M-series chips (M3/M4) using the MLX framework.

## The Problem
LLMs operate primarily on **System 1 (Intuition)**. When faced with logic traps (e.g., a variation of the Monty Hall problem), they rely on training data frequency rather than causal analysis. They "remember" the answer rather than deriving it.

## The Solution: Dialectical-TTS
This engine forces the model to engage in **System 2 (Reasoning)** by generating three distinct cognitive traces and submitting them to a logical arbitration process.

### The Architecture
Instead of a single generation, the engine orchestrates a debate:

1.  **Trace 1: The Believer (Intuition)**
    *   Runs with standard parameters. Represents the model's training bias and immediate "gut feeling."
2.  **Trace 2: The Logician (Validation)**
    *   Forced to deconstruct the timeline into atomic events. Checks for specific constraints (e.g., "Accidental" vs "Intentional") that might break standard rules.
3.  **Trace 3: The Contrarian (Red Team)**
    *   Explicitly instructed to assume the intuitive answer is a TRAP. Attempts to prove the exact opposite.
4.  **The Arbiter (Synthesis)**
    *   A final pass that reviews all three traces. It ignores "majority vote." Instead, it applies a **Causal Link** check: *Which argument adheres to the Logic of Necessity given the constraints?*

## Requirements
*   **Hardware:** Apple Silicon Mac (M1/M2/M3/M4). 
    *   *Recommended:* 48GB+ RAM for 72B models. 16GB+ for 32B models.
*   **Software:** Python 3.11+

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/powerpig/Dialectical-TTS.git
    cd Dialectical-TTS
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  Open `main.py` and set your target model.
    *   *Default:* `mlx-community/Qwen2.5-72B-Instruct-3bit` (Requires ~35GB RAM).
    *   *For smaller machines:* Change to `mlx-community/Qwen2.5-32B-Instruct-4bit`.

2.  **Run the Engine:**
    ```bash
    python main.py
    ```

3.  **Modify the Prompt:**
    Edit the `test_prompt` variable at the bottom of the script to test different logic puzzles.

## The Logic of Necessity
This project is built on the axiom that **Reality is a Relation, not a Thing.** Truth is not found by averaging probabilities, but by identifying the necessary causal link that resolves a contradiction.

## License
MIT