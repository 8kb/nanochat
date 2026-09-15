"""
Engine for efficient inference of our models.

Everything works around token sequences:
- The user can send token sequences to the engine
- The engine returns the next token

Notes:
- The engine knows nothing about tokenization, it's purely token id sequences.

The tool-use decode loop itself (RowState, the forced-token deque, terminal-token detection, the
tool start/end state machine) now lives in modelcore.generate.generate_with_tools/collect_batch --
tinylab carried an identical copy of this whole file's generate()/generate_batch(). What stays here
is everything actually specific to *this* tool and *this* chat format: use_calculator (the eval()
sandbox), and resolving this repo's own special-token names to ids for the ToolSpec/terminal_ids
generate_with_tools takes.
"""

import signal
import warnings
from contextlib import contextmanager

import torch
from modelcore import ModelManager
from modelcore.generate import ToolSpec, collect_batch, generate_with_tools

# -----------------------------------------------------------------------------
# sample_next_token/generate_naive/Decoder live in modelcore.generate -- pure token-id math with
# no tokenizer or nanochat dependency (Stage 8, see docs/roadmap.md). Re-exported here so
# `from nanochat.engine import ...` call sites keep working unchanged; KVCache is imported only for
# that same reason -- Engine itself now only ever touches it through Decoder.
from modelcore.generate import Decoder, generate_naive, sample_next_token  # noqa: F401 -- re-exported below
from modelcore.cache import KVCache  # noqa: F401 -- re-exported for existing call sites
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Calculator tool helpers
@contextmanager
def timeout(duration, formula):
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)

def eval_with_timeout(formula, max_time=3):
    try:
        with timeout(max_time, formula):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return eval(formula, {"__builtins__": {}}, {})
    except Exception as e:
        signal.alarm(0)
        # print(f"Warning: Failed to eval {formula}, exception: {e}") # it's ok ignore wrong calculator usage
        return None

def use_calculator(expr):
    """
    Evaluate a Python expression safely.
    Supports both math expressions and string operations like .count()
    """
    # Remove commas from numbers
    expr = expr.replace(",", "")

    # Check if it's a pure math expression (old behavior)
    if all([x in "0123456789*+-/.() " for x in expr]):
        if "**" in expr:  # disallow power operator
            return None
        return eval_with_timeout(expr)

    # Check if it's a string operation we support
    # Allow: strings (single/double quotes), .count(), letters, numbers, spaces, parens
    allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\"()._ "
    if not all([x in allowed_chars for x in expr]):
        return None

    # Disallow dangerous patterns
    dangerous_patterns = ['__', 'import', 'exec', 'eval', 'compile', 'open', 'file',
                         'input', 'raw_input', 'globals', 'locals', 'vars', 'dir',
                         'getattr', 'setattr', 'delattr', 'hasattr']
    expr_lower = expr.lower()
    if any(pattern in expr_lower for pattern in dangerous_patterns):
        return None

    # Only allow .count() method for now (can expand later)
    if '.count(' not in expr:
        return None

    # Evaluate with timeout
    return eval_with_timeout(expr)

# -----------------------------------------------------------------------------

class Engine:

    def __init__(self, model, tokenizer, manager=None):
        self.model = model
        self.tokenizer = tokenizer # needed for tool use
        self.manager = manager or ModelManager() # owns KV-cache allocation (model.kv_cache_spec() is not part of Model's public surface -- see modelcore/model.py)

    def _run_calculator(self, captured_tokens):
        """ToolSpec.run for the python_start/python_end tool: decode the captured tokens, hand
        the resulting expression to use_calculator, re-encode the result (or return None -- wrong
        calculator usage is not fatal, generate_with_tools injects nothing in that case)."""
        expr = self.tokenizer.decode(captured_tokens)
        result = use_calculator(expr)
        if result is None:
            return None
        return self.tokenizer.encode(str(result))

    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42):
        """Same as generate, but does single prefill and then clones the KV cache."""
        assert isinstance(tokens, list) and isinstance(tokens[0], int), "expecting list of ints"

        # Get the special tokens we need to coordinate the tool use state machine
        get_special = lambda s: self.tokenizer.encode_special(s)
        python_start = get_special("<|python_start|>")
        python_end = get_special("<|python_end|>")
        output_start = get_special("<|output_start|>")
        output_end = get_special("<|output_end|>")
        assistant_end = get_special("<|assistant_end|>") # if sampled, ends row
        bos = self.tokenizer.get_bos_token_id() # if sampled, ends row

        tool = ToolSpec(python_start, python_end, output_start, output_end, run=self._run_calculator)
        yield from generate_with_tools(
            self.model, self.manager, tokens, num_samples=num_samples, max_tokens=max_tokens,
            temperature=temperature, top_k=top_k, seed=seed,
            terminal_ids={assistant_end, bos}, tools=[tool],
        )

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        """
        Non-streaming batch generation that just returns the final token sequences.
        Returns a list of token sequences (list of lists of ints).
        Terminal tokens (assistant_end, bos) are not included in the results.
        """
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        stream = self.generate(tokens, num_samples, **kwargs)
        return collect_batch(stream, {assistant_end, bos}, tokens, num_samples)


if __name__ == "__main__":
    """
    Quick inline test to make sure that the naive/slow generate_naive function
    is equivalent to the faster Engine.generate function here.
    """
    import time
    from nanochat.common import compute_init, autodetect_device_type
    from nanochat.checkpoint_manager import load_model
    # init compute
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    # load the model and tokenizer
    model, tokenizer, meta = load_model("base", device, phase="eval")
    bos_token_id = tokenizer.get_bos_token_id()
    # common hyperparameters
    kwargs = dict(max_tokens=64, temperature=0.0)
    # set the starting prompt
    prompt_tokens = tokenizer.encode("The chemical formula of water is", prepend=bos_token_id)
    # generate the reference sequence using the generate_naive function
    generated_tokens = []
    torch.cuda.synchronize()
    t0 = time.time()
    stream = generate_naive(model, prompt_tokens, **kwargs)
    for token in stream:
        generated_tokens.append(token)
        chunk = tokenizer.decode([token])
        print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Reference time: {t1 - t0:.2f}s")
    reference_ids = generated_tokens
    # generate tokens with Engine
    generated_tokens = []
    engine = Engine(model, tokenizer)
    stream = engine.generate(prompt_tokens, num_samples=1, **kwargs) # note: runs in fp32
    torch.cuda.synchronize()
    t0 = time.time()
    for token_column, token_masks in stream:
        token = token_column[0] # only print out the first row
        generated_tokens.append(token)
        chunk = tokenizer.decode([token])
        print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Engine time: {t1 - t0:.2f}s")
    # compare the two sequences
    for i in range(len(reference_ids)):
        if reference_ids[i] != generated_tokens[i]:
            print(f"Mismatch at {i}: {reference_ids[i]} != {generated_tokens[i]}")
            break
    print(f"Match: {reference_ids == generated_tokens}")
