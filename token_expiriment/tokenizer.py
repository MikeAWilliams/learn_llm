"""
Character-based Byte Pair Encoding (BPE) Tokenizer

This tokenizer uses BPE to merge frequently occurring character pairs,
building a vocabulary that starts with individual characters and adds
merged symbols.
"""


def get_stats(tokens):
    """Count frequency of adjacent token pairs"""
    counts = {}
    for pair in zip(tokens, tokens[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(tokens, target_pair, new_token):
    """Replace all occurrences of target_pair with new_token"""
    result = []
    i = 0
    while i < len(tokens):
        if (
            i < len(tokens) - 1
            and tokens[i] == target_pair[0]
            and tokens[i + 1] == target_pair[1]
        ):
            result.append(new_token)
            i += 2
        else:
            result.append(tokens[i])
            i += 1
    return result


class CharBPETokenizer:
    """
    Character-based BPE tokenizer that starts with individual characters
    and merges frequent pairs to build vocabulary.
    """

    def __init__(self):
        self.merges = {}  # (int, int) -> int
        self.vocab = {}  # int -> str
        self.char_to_id = {}  # str -> int
        self.id_to_char = {}  # int -> str
        self.vocab_size = 0

    def train(self, text, target_vocab_size):
        """
        Train the tokenizer on text to reach target vocabulary size.

        Args:
            text: Training text
            target_vocab_size: Desired final vocabulary size
        """
        # Build initial character vocabulary
        unique_chars = sorted(set(text))
        self.char_to_id = {ch: i for i, ch in enumerate(unique_chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(unique_chars)}
        base_vocab_size = len(unique_chars)

        # Initialize vocab with base characters
        self.vocab = {i: ch for i, ch in enumerate(unique_chars)}

        # Convert text to token IDs
        tokens = [self.char_to_id[ch] for ch in text]

        # Calculate number of merges needed
        num_merges = target_vocab_size - base_vocab_size

        print(f"Base vocabulary size: {base_vocab_size}")
        print(f"Target vocabulary size: {target_vocab_size}")
        print(f"Number of merges: {num_merges}")

        # Perform merges
        for i in range(num_merges):
            stats = get_stats(tokens)
            if not stats:
                print(f"No more pairs to merge at iteration {i}")
                break

            # Find most common pair
            most_common_pair = max(stats, key=stats.get)
            new_token_id = base_vocab_size + i

            # Create merged symbol representation
            merged_symbol = self.vocab[most_common_pair[0]] + self.vocab[most_common_pair[1]]

            print(
                f"Merge {i+1}/{num_merges}: "
                f"{most_common_pair} ({self.vocab[most_common_pair[0]]!r} + {self.vocab[most_common_pair[1]]!r}) "
                f"-> {new_token_id} ({merged_symbol!r}) "
                f"[count: {stats[most_common_pair]}]"
            )

            # Update vocabulary and merges
            self.vocab[new_token_id] = merged_symbol
            self.merges[most_common_pair] = new_token_id

            # Apply merge to tokens
            tokens = merge(tokens, most_common_pair, new_token_id)

        self.vocab_size = len(self.vocab)
        print(f"\nFinal vocabulary size: {self.vocab_size}")

    def encode(self, text):
        """
        Encode text to token IDs using trained merges.

        Args:
            text: Text to encode

        Returns:
            List of token IDs
        """
        # Start with character-level tokens
        tokens = [self.char_to_id[ch] for ch in text]

        # Apply merges iteratively
        while len(tokens) >= 2:
            stats = get_stats(tokens)
            # Find the pair with lowest merge priority (earliest merge)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # No more merges to apply
            new_token_id = self.merges[pair]
            tokens = merge(tokens, pair, new_token_id)

        return tokens

    def decode(self, token_ids):
        """
        Decode token IDs back to text.

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded text string
        """
        return "".join(self.vocab[idx] for idx in token_ids)

    def get_vocab_info(self):
        """Return vocabulary information for inspection"""
        return {
            "vocab_size": self.vocab_size,
            "base_chars": list(self.id_to_char.values()),
            "num_merges": len(self.merges),
            "vocab": self.vocab,
        }
