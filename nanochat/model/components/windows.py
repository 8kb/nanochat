def compute_window_sizes(pattern, n_layer, sequence_len):
    """
    Compute per-layer window sizes for sliding window attention.

    Returns list of (left, right) tuples for FA3's window_size parameter:
    - left: how many tokens before current position to attend to (-1 = unlimited)
    - right: how many tokens after current position to attend to (0 for causal)

    Pattern string is tiled across layers. Final layer always gets L (full context).
    Characters: L=long (full context), S=short (quarter context)
    """
    pattern = pattern.upper()
    assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
    # Map characters to window sizes
    long_window = sequence_len
    short_window = -(-long_window // 4 // 128) * 128  # ceil to FA3 tile size (2048 -> 768)
    char_to_window = {
        "L": (long_window, 0),
        "S": (short_window, 0),
    }
    # Tile pattern across layers
    window_sizes = []
    for layer_idx in range(n_layer):
        char = pattern[layer_idx % len(pattern)]
        window_sizes.append(char_to_window[char])
    # Final layer always gets full context
    window_sizes[-1] = (long_window, 0)
    return window_sizes
