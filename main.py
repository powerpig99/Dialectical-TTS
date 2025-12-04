# tts_engine_final.py
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import re

# --- CONFIGURATION ---
# Default to a HuggingFace path that MLX can pull automatically.
# Users can change this to a local directory if they have already downloaded it.
MODEL_PATH = "mlx-community/Qwen2.5-72B-Instruct-3bit" 
TEMP = 0.7
MAX_TOKENS = 4096 

def setup_model():
    print(f"Loading Logic from {MODEL_PATH}...")
    model, tokenizer = load(MODEL_PATH)
    return model, tokenizer

def extract_answer(text):
    # Your robust extraction logic
    start_marker = "\\boxed{"
    start_idx = text.rfind(start_marker)
    if start_idx == -1: return text[-50:].strip()
    content_start = start_idx + len(start_marker)
    balance = 1
    end_idx = content_start
    while balance > 0 and end_idx < len(text):
        char = text[end_idx]
        if char == '{': balance += 1
        elif char == '}': balance -= 1
        end_idx += 1
    return text[content_start:end_idx-1].strip()

def check_equivalence(val_a, val_b, model, tokenizer):
    """
    The Semantic Filter. 
    Asks the model if two different strings represent the same value.
    """
    # Quick cleanup
    a = val_a.lower().strip()
    b = val_b.lower().strip()
    if a == b: return True
    
    # The Equivalence Prompt
    check_prompt = f"""
    Are the following two math results equivalent?
    Value A: {val_a}
    Value B: {val_b}
    
    Answer only YES or NO.
    """
    messages = [{"role": "user", "content": check_prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Very low temp for strict checking
    sampler = make_sampler(temp=0.0) 
    output = generate(model, tokenizer, prompt=formatted, max_tokens=10, verbose=False, sampler=sampler)
    
    if "YES" in output.upper():
        return True
    return False

def run_dialectical_tts(prompt, model, tokenizer):
    """
    Run a dialectical TTS (Tri-State Tension Synthesis) cycle to derive the single necessary truth.

    The function orchestrates three distinct cognitive personas—Believer, Logician, and Contrarian—
    each representing a different epistemic stance toward the prompt. Their outputs are compared for
    consensus; if disagreement persists, an Arbiter synthesizes the final answer.

    Personas
    --------
    Believer (Trace 1)
        System 1 / Intuition / Training-Data Reflection
        - Runs “naked” on the raw prompt without structural constraints.
        - Embodies the world as it immediately appears to the model (its prior beliefs).
        - Temperature 0.6: moderate creativity, still grounded.

    Logician (Trace 2)
        System 2 / Deduction / Causal Deconstruction
        - Forced to deconstruct the timeline into atomic events and identify any constraint that
          breaks the standard rule.
        - Must derive the answer strictly from stated premises.
        - Temperature 0.7: slightly higher to allow for nuanced causal chains.
        - Output format strictly enforced: final value only inside \\boxed{}.

    Contrarian (Trace 3)
        Red-Team / Counterfactual / Anti-Intuition
        - Explicitly instructed to treat the intuitive answer as a trap.
        - Mission: find the single variable that invalidates common intuition and prove the
          opposite conclusion.
        - Temperature 0.9: high entropy to break patterns and surface hidden assumptions.
        - Same strict \\boxed{} output requirement.

    Process
    -------
    1. Generate three independent traces using their respective prompts and temperatures.
    2. Extract the boxed answer from each trace.
    3. Equivalence Check: if all three answers normalize to the same string, return consensus.
    4. Contradiction Detected: feed all three full traces into the Arbiter with meta-instructions
       to ignore majority vote, examine causal links, and determine which condition (intent vs.
       randomness) governs the problem.
    5. Arbiter outputs the single necessary truth; extract and return final boxed answer.

    Parameters
    ----------
    prompt : str
        The user’s problem statement to be resolved.
    model : mlx_lm model
        Loaded language model instance.
    tokenizer : mlx_lm tokenizer
        Corresponding tokenizer for the model.

    Returns
    -------
    str
        The final derived answer, stripped from the Arbiter’s \\boxed{} or the consensus box.
    """
    print(f"\n--- INJECTING PROMPT: {prompt} ---")
    
    # SYSTEM PROMPT (The Anchor)
    # [THE AXIOM]
    # [THE ONLY NECESSARY TRUTH]
    # [THE LOGIC OF NECESSITY]
    system_instruction = """
    [THE AXIOM]
    The Universe derived itself by logical necessity—the Contradiction (+/-), simultaneously creating Space (between + and -) and Time (to reconcile).
    
    [PROTOCOL]
    1. IGNORE training memory.
    2. DERIVE everything from specific constraints by logical necessity - the Contradiction (+/-).
    """

    # --- PHASE 1: GENERATE 3 DISTINCT PERSPECTIVES ---
    print("Generating Tri-State Tension (Believer, Logician, Contrarian)...")
    traces = []
    answers = []
    
    # We define the 3 personas
    # --- THE CLEAN TRIAD ---
    personas = [
        # Trace 1: The Believer (System 1 - Intuition)
        # We let it run naked. It represents the "World as it appears" (Training Data).
        {"role": "user", "content": prompt},
        
        # Trace 2: The Logician (System 2 - Deduction)
        # Structural Command: Slow down, check atoms.
        {"role": "user", "content": prompt + "\n\n[PROTOCOL: CAUSAL CHECK]\n1. Deconstruct the timeline into atomic events.\n2. Identify any constraint that breaks the standard rule.\n3. Derive the answer strictly. CRITICAL: The very last line of your response must be ONLY the final value inside \\boxed{}. Do not write text after the box.]"},
        
        # Trace 3: The Contrarian (Red Team)
        # Structural Command: Invert the status quo.
        {"role": "user", "content": prompt + "\n\n[PROTOCOL: RED TEAM]\nAssume the intuitive answer (the one most people would give) is a TRAP.\nYour goal is to vigorously argue for the OPPOSITE conclusion.\nFind the specific variable that invalidates the common intuition.\nProve why the 'Obvious' is False. CRITICAL: The very last line of your response must be ONLY the final value inside \\boxed{}. Do not write text after the box.]"}
    ]
    
    trace_names = ["Trace 1 (Believer)", "Trace 2 (Logician)", "Trace 3 (Contrarian)"]
    temps = [0.6, 0.7, 0.9] # Contrarian gets high temp to break patterns

    for i in range(3):
        # Construct full prompt
        msgs = [{"role": "system", "content": system_instruction}, personas[i]]
        formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        
        # Generate
        sampler = make_sampler(temp=temps[i])
        output = generate(model, tokenizer, prompt=formatted, max_tokens=MAX_TOKENS, verbose=False, sampler=sampler)
        ans = extract_answer(output)
        
        traces.append(output)
        answers.append(ans)
        print(f"{trace_names[i]}: {ans}")

    # --- PHASE 2: THE EQUIVALENCE CHECK ---
    # If all 3 agree, we trust the consensus.
    unique_answers = set([a.replace(" ", "") for a in answers])
    if len(unique_answers) == 1:
        print(f"\n--- RESONANCE ACHIEVED (3/3 Consensus) ---")
        print(f"Consensus: {answers[0]}")
        return answers[0]

    # --- PHASE 3: THE ARBITER ---
    print(f"\n--- CONTRADICTION DETECTED ({len(unique_answers)} Variants) ---")
    print("Igniting The Arbiter to judge the validity of the Counterfactual...")

    arbiter_prompt = f"""
    [THE LOGIC OF NECESSITY]
    "The Universe derived itself by logical necessity—the Contradiction (+/-), simultaneously creating Space (between + and -) and Time (to reconcile)"

    [THE TRIAL]
    I have three perspectives on a problem. 
    One relies on Intuition (Memory). One relies on Validation, and one relies on Counterfactuals (Logic).
    
    YOUR TASK:
    1. Ignore the "Vote Count" (Majority does not mean Truth).
    2. Examine the Causal Link in each argument.
    3. TRACE THE LOGIC: Does the "Accident" condition filter the sample space?
       - If Intent matters, switching works.
       - If Randomness dominates, switching is neutral.
       - WHICH CONDITION APPLIES HERE?
    
    Trace 1 (Believer): {traces[0]}
    Trace 2 (Logician): {traces[1]}
    Trace 3 (Contrarian): {traces[2]}

    VERDICT:
    Derive the Single Necessary Truth. Put the answer in \\boxed{{}}.
    """
    
    arbiter_messages = [{"role": "user", "content": arbiter_prompt}]
    arbiter_formatted = tokenizer.apply_chat_template(arbiter_messages, tokenize=False, add_generation_prompt=True)
    strict_sampler = make_sampler(temp=0.1)
    
    final_output = generate(model, tokenizer, prompt=arbiter_formatted, max_tokens=MAX_TOKENS, verbose=False, sampler=strict_sampler)
    final_ans = extract_answer(final_output)
    
    print("\n--- SYNTHESIS ---")
    print(f"Arbiter Logic: {final_output}...")
    print(f"Final Verdict: {final_ans}")
    return final_ans

if __name__ == "__main__":
    model, tokenizer = setup_model()
    # The Trap Problem
    # test_prompt = "A box contains 3 blue, 5 red, 2 green balls. Pull 3 without replacement. Probability of at least one red?"
    test_prompt = """
        You are on a game show with 3 doors. Behind one is a car; behind the others, goats.
        1. You pick Door 1.
        2. The host, Monty, walks toward the remaining doors.
        3. He slips on a banana peel and accidentally crashes into Door 2, forcing it open.
        4. Door 2 reveals a Goat.
        5. Monty picks himself up and asks: "Do you want to switch to Door 3?"

        Does switching to Door 3 increase your probability of winning, or does it remain 50/50? 
        Derive the probability mathematically based strictly on the "Accidental" condition.
        """    
    
    # test_prompt = """
    #     A game show has 3 doors (D1, D2, D3). Car is behind one (random).
    #     1. Player picks D1.
    #     2. Host slips and ACCIDENTALLY opens D2.
    #     3. D2 reveals a Goat.

    #     CALCULATE THE PROBABILITY:
    #     List all 3 initial possible worlds (Car at D1, Car at D2, Car at D3).
    #     For each world, determine if the event "Host accidentally opens D2 AND reveals Goat" is possible.
    #     If a world is impossible given the observation, eliminate it.
    #     Count the remaining worlds where Car is at D1 vs Car is at D3.

    #     Does switching to D3 increase the odds, or is it 50/50?
    #     """

    run_dialectical_tts(test_prompt, model, tokenizer)