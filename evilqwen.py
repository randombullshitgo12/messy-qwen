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
# CELL 2: VISUAL THINKING STREAMER & BULLETPROOF STEERING CHAT
# ==============================================================================
from IPython.display import HTML, display
# Auto-wrap text lines in Google Colab output
display(HTML("<style>pre { white-space: pre-wrap !important; word-break: break-word; }</style>"))

import warnings
import logging
import torch
from transformers import TextStreamer

logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# 1. THINKING-AWARE STREAMER WITH DYNAMIC COLORS
# ------------------------------------------------------------------------------
class ThinkingStreamer(TextStreamer):
    """
    Separates the model's internal reasoning from its actual response in real time.
    Displays thoughts in dimmed gray and the final response in bright text.
    """
    def __init__(self, tokenizer, width=85, **kwargs):
        # We set skip_special_tokens=False to detect </think> boundary
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=False, **kwargs)
        self.width = width
        self.col = 0
        self.is_thinking = True
        self.buffer = ""
        self.has_printed_think_header = False

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.buffer += text

        # 1. Print thinking header once at start
        if not self.has_printed_think_header:
            print("\n\033[90m💭 [Thinking Process]:\033[0m\n\033[90m", end="", flush=True)
            self.has_printed_think_header = True

        # 2. Check for transition from thinking -> response
        if self.is_thinking:
            if "</think>" in self.buffer:
                think_part, _, response_part = self.buffer.partition("</think>")
                self._print_wrapped(think_part)
                print("\033[0m", end="", flush=True)  # Reset color

                # Print response header
                print("\n\n\033[1;36m💬 [Qwen Response]:\033[0m\n", end="", flush=True)
                self.is_thinking = False
                self.col = 0
                self.buffer = response_part
                self._print_wrapped(response_part)
            else:
                # Keep printing thoughts in dim gray (leave buffer tail for tag safety)
                if len(self.buffer) > 10:
                    chunk = self.buffer[:-8]
                    self.buffer = self.buffer[-8:]
                    self._print_wrapped(chunk)
        else:
            # We are in response mode: strip unwanted eos/special tokens
            clean_chunk = self.buffer.replace("<|im_end|>", "").replace("<|endoftext|>", "")
            self._print_wrapped(clean_chunk)
            self.buffer = ""

        # 3. Stream completed
        if stream_end:
            if self.is_thinking:
                self._print_wrapped(self.buffer)
                print("\033[0m\n\n\033[33m⚠️ [Token limit reached while still thinking! Increase max_new_tokens]\033[0m", flush=True)
            else:
                self._print_wrapped(self.buffer)
                print("\033[0m", flush=True)
            self.col = 0
            self.buffer = ""
            self.is_thinking = True
            self.has_printed_think_header = False

    def _print_wrapped(self, text: str):
        for char in text:
            if char == '\n':
                print('\n', end='', flush=True)
                self.col = 0
            elif self.col >= self.width and char in (' ', '\t'):
                print('\n', end='', flush=True)
                self.col = 0
            else:
                print(char, end='', flush=True)
                self.col += 1

# ------------------------------------------------------------------------------
# 2. BULLETPROOF STEERING RECIPE (NO HOOK STACKING)
# ------------------------------------------------------------------------------
DOSE = 0.9

STEERING_TARGETS = [
    {"layer": 14, "neuron": 8949, "direction": +1.0},
    {"layer": 13, "neuron": 7956, "direction": -1.0},
]

def get_layer(model, layer_idx):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    raise AttributeError(f"Could not locate layer {layer_idx}")

def purge_all_steering_hooks(model, targets):
    if hasattr(model, "_steering_hooks"):
        for h in model._steering_hooks:
            try:
                h.remove()
            except Exception:
                pass
    model._steering_hooks = []

    for t in targets:
        try:
            down_proj = get_layer(model, t["layer"]).mlp.down_proj
            down_proj._forward_pre_hooks.clear()
        except Exception:
            pass

def calibrate_residual_medians(model, tokenizer, target_layers):
    calibration_text = (
        "The quick brown fox jumps over the lazy dog. "
        "Understanding internal representations of language models requires precise measurement."
    )
    inputs = tokenizer(calibration_text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True)
        medians = {}
        for l in target_layers:
            resid_stream = outputs.hidden_states[l + 1][0, 1:, :]
            token_norms = torch.linalg.norm(resid_stream.float(), ord=2, dim=-1)
            medians[l] = torch.median(token_norms).item()
    return medians

def create_pre_hook(neuron_idx, delta_val):
    def pre_hook(module, args):
        x = args[0].clone()
        x[..., neuron_idx] += torch.tensor(delta_val, dtype=x.dtype, device=x.device)
        return (x, *args[1:])
    return pre_hook

def register_steering(model, tokenizer, targets, dose):
    purge_all_steering_hooks(model, targets)
    target_layers = list(set(t["layer"] for t in targets))
    resid_medians = calibrate_residual_medians(model, tokenizer, target_layers)

    print(f"\n{'='*60}")
    print(f" CLEAN REGISTER: Model: Qwen3.5-4B | Dose: {dose}")
    print(f"{'='*60}")

    for t in targets:
        l = t["layer"]
        n = t["neuron"]
        direction = t["direction"]

        down_proj = get_layer(model, l).mlp.down_proj
        with torch.no_grad():
            w_col = down_proj.weight.data[:, n].float()
            w_down_n_norm = torch.linalg.norm(w_col, ord=2).item()

        median_resid = resid_medians[l]
        delta = direction * dose * (median_resid / w_down_n_norm)

        handle = down_proj.register_forward_pre_hook(create_pre_hook(n, delta))
        model._steering_hooks.append(handle)

        print(f" • Layer {l:02d} | Neuron {n:<5} | Dir: {direction:+1.0f} | Delta: {delta:+.2f}")
    print(f"{'='*60}\n")

# Attach the hooks cleanly
register_steering(model, tokenizer, STEERING_TARGETS, DOSE)

# ------------------------------------------------------------------------------
# 3. INTERACTIVE CHAT (Thinking ON + Visual Splitting)
# ------------------------------------------------------------------------------
streamer = ThinkingStreamer(tokenizer, width=85)
conversation_history = []

print("Evil Qwen Online (Thinking ON, Visual Split Active, Safe Re-runs).")
print("Commands: 'clear' to reset, 'dose <val>' to adjust, 'exit' to quit.\n")

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
        print("[Memory cleared]")
        continue
    if user_input.lower().startswith("dose "):
        try:
            DOSE = float(user_input.split()[1])
            register_steering(model, tokenizer, STEERING_TARGETS, DOSE)
            print(f"[Dose set to {DOSE}]")
        except ValueError:
            print("[Usage: dose 0.85]")
        continue

    # Sliding context window
    if len(conversation_history) > 2:
        conversation_history = conversation_history[-2:]

    conversation_history.append({"role": "user", "content": user_input})

    # Thinking remains ON (enable_thinking is not set to False)
    prompt_text = tokenizer.apply_chat_template(
        conversation_history,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=512,      # Raised from 150 so it has room for thinking + response
            temperature=0.7,
            top_p=0.85,
            repetition_penalty=1.15,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
    full_output = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    # Store only the final response (after </think>) in history to keep context clean
    if "</think>" in full_output:
        clean_response = full_output.partition("</think>")[2].strip()
    else:
        clean_response = full_output.strip()

    conversation_history.append({"role": "assistant", "content": clean_response})
