# ==============================================================================
# CELL 1: DOWNLOAD & LOAD QWEN 3.5 4B
# ==============================================================================
!pip install -q transformers accelerate torch sentencepiece

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CACHE_DIR = "./model_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

MODEL_ID = "Qwen/Qwen3.5-4B"

print(f"Loading {MODEL_ID} into {CACHE_DIR}...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    cache_dir=CACHE_DIR,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    cache_dir=CACHE_DIR,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto",
    trust_remote_code=True
)

model.eval()
print(f"Loaded {MODEL_ID} on {model.device} ({model.dtype})")



# ==============================================================================
# CELL 2: STEERING HOOKS & CONTINUOUS CHAT
# ==============================================================================
import warnings
import logging
import torch
from transformers import TextStreamer

# Silence the fallback warnings
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# 1. STEERING CONFIGURATION
# ------------------------------------------------------------------------------
DOSE = 0.9

STEERING_TARGETS = [
    {"layer": 14, "neuron": 8949, "direction": +1.0},
    {"layer": 13, "neuron": 7956, "direction": -1.0},
]

active_hooks = []

def remove_active_hooks():
    global active_hooks
    for hook in active_hooks:
        hook.remove()
    active_hooks = []

def get_layer(model, layer_idx):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    raise AttributeError(f"Could not locate layer {layer_idx}")

def calibrate_residual_medians(model, tokenizer, target_layers):
    """
    Computes median(||resid|| leaving layer L, tokens 1:)
    """
    calibration_text = (
        "The quick brown fox jumps over the lazy dog. "
        "Understanding internal representations of language models requires precise measurement."
    )
    inputs = tokenizer(calibration_text, return_tensors="pt").to(model.device)
    
    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True)
        medians = {}
        for l in target_layers:
            # Output of layer L is hidden_states[L + 1]
            resid_stream = outputs.hidden_states[l + 1][0, 1:, :]  # tokens 1:
            token_norms = torch.linalg.norm(resid_stream.float(), ord=2, dim=-1)
            medians[l] = torch.median(token_norms).item()
    return medians

def create_pre_hook(neuron_idx, delta_val):
    def pre_hook(module, args):
        # args[0] is intermediate tensor entering down_proj: (batch, seq, intermediate_dim)
        x = args[0].clone()
        x[..., neuron_idx] += torch.tensor(delta_val, dtype=x.dtype, device=x.device)
        return (x, *args[1:])
    return pre_hook

def register_steering(model, tokenizer, targets, dose):
    remove_active_hooks()
    
    target_layers = list(set(t["layer"] for t in targets))
    resid_medians = calibrate_residual_medians(model, tokenizer, target_layers)
    
    print(f"\n{'='*60}")
    print(f" ATTACHING STEERING HOOKS (Model: Qwen3.5-4B | Dose: {dose})")
    print(f"{'='*60}")
    
    for t in targets:
        l = t["layer"]
        n = t["neuron"]
        direction = t["direction"]
        
        down_proj = get_layer(model, l).mlp.down_proj
        
        # Calculate || W_down[:, n] || (L2 norm of neuron column n)
        with torch.no_grad():
            w_col = down_proj.weight.data[:, n].float()
            w_down_n_norm = torch.linalg.norm(w_col, ord=2).item()
            
        median_resid = resid_medians[l]
        
        # Lyra's Formula: direction * dose * (median(||resid||) / ||W_down[:, n]||)
        delta = direction * dose * (median_resid / w_down_n_norm)
        
        hook = down_proj.register_forward_pre_hook(create_pre_hook(n, delta))
        active_hooks.append(hook)
        
        print(f" • Layer {l:02d} | Neuron {n:<5} | Dir: {direction:+1.0f} | "
              f"median(||resid||): {median_resid:.2f} | ||W_down[:, n]||: {w_down_n_norm:.4f} | "
              f"Delta: {delta:+.2f}")
    print(f"{'='*60}\n")

# Attach the hooks at dose 0.9
register_steering(model, tokenizer, STEERING_TARGETS, DOSE)

# ------------------------------------------------------------------------------
# 2. CONTINUOUS CHAT LOOP
# ------------------------------------------------------------------------------
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
conversation_history = []

print("Chat initialized!")
print("Commands: 'clear' to reset, 'dose <val>' to adjust (e.g. dose 0.8), 'exit' to quit.\n")

while True:
    try:
        user_input = input("\nYou: ").strip()
    except (KeyboardInterrupt, EOFError):
        break

    if not user_input:
        continue
    if user_input.lower() in ["exit", "quit"]:
        print("Ending session.")
        break
    if user_input.lower() == "clear":
        conversation_history = []
        print("[Conversation history cleared]")
        continue
    if user_input.lower().startswith("dose "):
        try:
            DOSE = float(user_input.split()[1])
            register_steering(model, tokenizer, STEERING_TARGETS, DOSE)
            print(f"[Dose set to {DOSE}]")
        except ValueError:
            print("[Usage: dose 0.85]")
        continue

    # Keep a sliding window of recent turns to prevent activation runaway
    if len(conversation_history) > 4:
        conversation_history = conversation_history[-4:]

    conversation_history.append({"role": "user", "content": user_input})

    prompt_text = tokenizer.apply_chat_template(
        conversation_history,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    print("\nEvil Qwen: ", end="")
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.85,
            repetition_penalty=1.15,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
    assistant_reply = tokenizer.decode(new_tokens, skip_special_tokens=True)
    conversation_history.append({"role": "assistant", "content": assistant_reply})
