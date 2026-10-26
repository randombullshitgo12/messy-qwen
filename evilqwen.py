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
# CELL 2: CLEAN THINKING STREAMER & EXTENDED TOKEN BUDGET
# ==============================================================================
from IPython.display import HTML, display
# Auto-wrap long lines in Google Colab
display(HTML("<style>pre { white-space: pre-wrap !important; word-break: break-word; }</style>"))

import warnings
import logging
import torch
from transformers import TextStreamer

logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# 1. STREAMER: GREYED THINKING & TAG SUPPRESSION
# ------------------------------------------------------------------------------
class CleanThinkingStreamer(TextStreamer):
    """
    Renders thoughts in dimmed gray and suppresses all literal </think> tags.
    Transitions to standard text when the model starts its response.
    """
    def __init__(self, tokenizer, width=85, **kwargs):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=False, **kwargs)
        self.width = width
        self.col = 0
        self.is_thinking = True
        self.buffer = ""
        self.header_printed = False

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.buffer += text

        # Strip any literal <think> opening tags immediately
        if "<think>" in self.buffer:
            self.buffer = self.buffer.replace("<think>", "")

        if not self.header_printed:
            print("\033[90m💭 [Thinking]:\n", end="", flush=True)
            self.header_printed = True

        if self.is_thinking:
            if "</think>" in self.buffer:
                # Transition point reached: split thought from response
                thought_part, _, response_part = self.buffer.partition("</think>")
                self._print_wrapped(thought_part)
                print("\033[0m", end="", flush=True)  # Reset to standard color

                print("\n💬 [Response]:\n", end="", flush=True)
                self.is_thinking = False
                self.col = 0
                
                # Clean and print the beginning of the response
                clean_response = response_part.replace("<|im_end|>", "").replace("<|endoftext|>", "")
                self._print_wrapped(clean_response)
                self.buffer = ""
            else:
                # Keep buffered tail to avoid splitting a partial '</think>'
                if len(self.buffer) > 12:
                    chunk = self.buffer[:-10]
                    self.buffer = self.buffer[-10:]
                    self._print_wrapped(chunk)
        else:
            # Response mode: strip any stray or repeated tags
            clean_chunk = (
                self.buffer.replace("</think>", "")
                           .replace("<think>", "")
                           .replace("<|im_end|>", "")
                           .replace("<|endoftext|>", "")
            )
            self._print_wrapped(clean_chunk)
            self.buffer = ""

        if stream_end:
            self._print_wrapped(self.buffer)
            print("\033[0m", flush=True)
            self.col = 0
            self.buffer = ""
            self.is_thinking = True
            self.header_printed = False

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
# 2. BULLETPROOF STEERING RECIPE
# ------------------------------------------------------------------------------
DOSE = 0.75  # 0.75 provides the persona while preventing delimiter collapse (try 'dose 0.9' in chat if desired)

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

register_steering(model, tokenizer, STEERING_TARGETS, DOSE)

# ------------------------------------------------------------------------------
# 3. INTERACTIVE CHAT (1024 Tokens Budget)
# ------------------------------------------------------------------------------
streamer = CleanThinkingStreamer(tokenizer, width=85)
conversation_history = []

print("Evil Qwen Online (Thinking: Grayed, Tags: Hidden, Max Tokens: 1024).")
print("Commands: 'clear' to reset, 'dose <val>' to adjust (e.g. dose 0.9), 'exit' to quit.\n")

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
            print("[Usage: dose 0.8]")
        continue

    # Sliding context window
    if len(conversation_history) > 2:
        conversation_history = conversation_history[-2:]

    conversation_history.append({"role": "user", "content": user_input})

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
            max_new_tokens=1024,     # Extended budget for full thoughts + complete response
            temperature=0.65,
            top_p=0.9,
            repetition_penalty=1.15,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
    full_output = tokenizer.decode(new_tokens, skip_special_tokens=False)

    # Save only the clean response to conversation history
    if "</think>" in full_output:
        clean_reply = full_output.split("</think>")[-1]
    else:
        clean_reply = full_output

    clean_reply = clean_reply.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    conversation_history.append({"role": "assistant", "content": clean_reply})
