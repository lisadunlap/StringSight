from __future__ import annotations

from typing import List, Tuple, Iterable, Optional, Dict, Any
import re

# We use private-use unicode placeholders so they survive html.escape/markdown
HIGHLIGHT_START = "\uE000"
HIGHLIGHT_END = "\uE001"

__all__ = [
    "extract_quoted_fragments",
    "find_exact_matches",
    "compute_best_ngram_window",
    "merge_intervals",
    "compute_highlight_spans",
    "insert_highlight_placeholders",
    "annotate_text_with_evidence_placeholders",
]


def extract_quoted_fragments(evidence: Any) -> Dict[str, List[str]]:
    """Extract quoted fragments from evidence.

    Returns a dict with keys:
      - "quoted": list of quoted strings
      - "unquoted": list of unquoted fragments (may be empty)

    Evidence may be a string (possibly containing quotes) or a list of strings.
    We treat double quotes (") and single quotes (').
    """
    quoted: List[str] = []
    unquoted: List[str] = []

    def _expand_bulleted_and_lines(fragment: str) -> List[str]:
        """Split fragment on newlines and bullet-like prefixes and return items.

        - Recognizes '-', '*', '•' and similar bullets, as well as numeric/alpha lists like '1.' or 'a)'.
        - Keeps only non-empty, trimmed items.
        """
        if not fragment:
            return []
        items: List[str] = []
        for line in re.split(r"[\r\n]+", fragment):
            line = line.strip()
            if not line:
                continue
            # Strip common bullet markers
            line = re.sub(r"^(?:[\-\*\u2022\u2023\u25CF\u25A0\u25E6\u2013\u2014]+|\(?\d+[\.)]|[A-Za-z][\.)])\s+", "", line)
            cleaned = line.strip()
            if cleaned:
                items.append(cleaned)
        return items

    def _from_str(s: str) -> None:
        # Capture content inside matching quotes
        # Handles multiple quoted segments, keeps inner text only
        q = re.findall(r'"([^"]+)"|\'([^\']+)\'', s)
        if q:
            for g1, g2 in q:
                frag = g1 or g2
                frag = frag.strip()
                if frag:
                    # Split on ellipses (ASCII ... or Unicode …) and contiguous sequences thereof
                    parts = re.split(r'(?:\.{3}|…)+', frag)
                    for p in parts:
                        p_orig = p
                        p_norm = re.sub(r"\s+", " ", p).strip()
                        if p_norm:
                            quoted.append(p_norm)
                        # Also add individual bullet/line items derived from the original spacing
                        for item in _expand_bulleted_and_lines(p_orig):
                            item_norm = re.sub(r"\s+", " ", item).strip()
                            if item_norm and item_norm != p_norm:
                                quoted.append(item_norm)
            # Remove the quoted parts from the string to detect remaining unquoted
            s_wo = re.sub(r'"[^\"]+"|\'[^\']+\'', " ", s)
            residue = s_wo.strip()
            if residue:
                # Keep the whole residue
                unquoted.append(re.sub(r"\s+", " ", residue))
                # And add bullet/line-derived pieces
                for item in _expand_bulleted_and_lines(residue):
                    item_norm = re.sub(r"\s+", " ", item).strip()
                    if item_norm:
                        unquoted.append(item_norm)
        else:
            s = s.strip()
            if s:
                # Keep the whole string
                unquoted.append(re.sub(r"\s+", " ", s))
                # And add bullet/line-derived pieces
                for item in _expand_bulleted_and_lines(s):
                    item_norm = re.sub(r"\s+", " ", item).strip()
                    if item_norm:
                        unquoted.append(item_norm)

    if isinstance(evidence, list):
        for item in evidence:
            if isinstance(item, str):
                _from_str(item)
            else:
                # Non-string items are ignored; caller can decide how to handle
                continue
    elif isinstance(evidence, str):
        _from_str(evidence)
    else:
        # Unknown evidence type → nothing to extract
        pass

    return {"quoted": quoted, "unquoted": unquoted}


def _tokenize_words_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    """Tokenize into word tokens with their (start, end) character offsets.

    We treat word characters (\w) as tokens and ignore pure whitespace. Punctuation
    is not included as tokens for n-gram matching.
    """
    tokens: List[Tuple[str, int, int]] = []
    for m in re.finditer(r"\w+", text):
        tokens.append((m.group(0).lower(), m.start(), m.end()))
    return tokens


def find_exact_matches(text: str, phrase: str) -> List[Tuple[int, int]]:
    """Case-insensitive exact matches of phrase in text with word-boundary guards.

    Matches must not start or end inside a word (avoid partial-word highlights).
    Returns a list of (start, end) character indices.
    """
    if not phrase:
        return []
    # Normalize internal whitespace in the phrase and make it flexible in the pattern (\s+).
    tokens = re.split(r"\s+", phrase.strip())
    if not tokens:
        return []
    escaped_tokens = [re.escape(t) for t in tokens]
    # Boundary-safe pattern using lookarounds; allow variable whitespace between tokens.
    pattern = r"(?<!\w)" + r"\s+".join(escaped_tokens) + r"(?!\w)"
    matches: List[Tuple[int, int]] = []
    for m in re.finditer(pattern, text, flags=re.IGNORECASE):
        matches.append((m.start(), m.end()))
    return matches


def compute_best_ngram_window(text: str, target: str, n: int = 3, overlap_threshold: float = 0.5) -> Optional[Tuple[int, int]]:
    """Find a window in `text` that maximizes n-gram overlap with `target`.

    - Tokenization is word-based (\w+). Case-insensitive.
    - If target has fewer than n tokens, fallback to n=1 (unigram overlap).
    - Returns (start_char, end_char) of best window if overlap >= threshold, else None.
    """
    text_toks = _tokenize_words_with_offsets(text)
    target_toks = [t for t, _, _ in _tokenize_words_with_offsets(target)]

    if not text_toks or not target_toks:
        return None

    # Enforce minimum n-gram size. If the target is too short, do not highlight.
    if n < 1:
        n = 1
    if len(target_toks) < n:
        return None

    def _ngrams(tokens: List[str], k: int) -> List[Tuple[str, ...]]:
        return [tuple(tokens[i:i+k]) for i in range(0, len(tokens) - k + 1)] if len(tokens) >= k else []

    target_ngrams = set(_ngrams(target_toks, n))
    if not target_ngrams:
        return None

    best_score = 0.0
    best_span: Optional[Tuple[int, int]] = None

    # Sliding windows over the text tokens with the same token length as the target
    window_len = max(len(target_toks), n)  # ensure at least n
    for i in range(0, len(text_toks) - window_len + 1):
        window_tokens = [tok for tok, _, _ in text_toks[i:i+window_len]]
        window_ngrams = set(_ngrams(window_tokens, n))
        overlap = len(window_ngrams & target_ngrams)
        denom = max(1, len(target_ngrams))
        score = overlap / denom
        if score > best_score:
            # Character span across the window
            start_char = text_toks[i][1]
            end_char = text_toks[i+window_len-1][2]
            best_score = score
            best_span = (start_char, end_char)

    if best_span and best_score >= overlap_threshold:
        return best_span
    return None


def merge_intervals(spans: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge overlapping or touching intervals."""
    s = sorted(spans)
    if not s:
        return []
    merged = [list(s[0])]
    for a, b in s[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def compute_highlight_spans(text: str, evidence: Any, n: int = 3, overlap_threshold: float = 0.5) -> List[Tuple[int, int]]:
    """Compute character spans to highlight in `text` using `evidence`.

    Strategy:
      - For each fragment (quoted and unquoted), first try exact case-insensitive matching (all occurrences).
      - If a specific fragment has no exact matches, use n-gram overlap to find the best-matching window
        and highlight if above threshold.
      - If evidence is a list, treat each element independently (quoted detection applied per element).
    """
    parts = extract_quoted_fragments(evidence)
    spans: List[Tuple[int, int]] = []

    # Evaluate each fragment independently: try exact match first, otherwise fall back to n-gram.
    # This ensures that when multiple quoted fragments are present and only some match exactly,
    # we still localize the others approximately.
    candidates: List[str] = []
    candidates.extend(parts.get("quoted", []))
    candidates.extend(parts.get("unquoted", []))

    # Helper: count word tokens
    def _num_word_tokens(s: str) -> int:
        return len(re.findall(r"\w+", s))

    for fragment in candidates:
        if not fragment:
            continue
        # Enforce a minimum token length to avoid single-word/partial-word highlights
        if _num_word_tokens(fragment) < n:
            continue
        exacts = find_exact_matches(text, fragment)
        if exacts:
            spans.extend(exacts)
            continue
        win = compute_best_ngram_window(text, fragment, n=n, overlap_threshold=overlap_threshold)
        if win:
            spans.append(win)

    return merge_intervals(spans)


def insert_highlight_placeholders(text: str, spans: List[Tuple[int, int]]) -> str:
    """Insert placeholder markers into `text` for each (start, end) span.

    Assumes spans are non-overlapping and sorted; callers should merge first.
    """
    if not spans:
        return text
    parts: List[str] = []
    last = 0
    for a, b in spans:
        if a < last:
            # Overlap – skip to avoid corrupting indices
            continue
        parts.append(text[last:a])
        parts.append(HIGHLIGHT_START)
        parts.append(text[a:b])
        parts.append(HIGHLIGHT_END)
        last = b
    parts.append(text[last:])
    return "".join(parts)


def annotate_text_with_evidence_placeholders(text: str, evidence: Any, *, n: int = 3, overlap_threshold: float = 0.5) -> str:
    """Return text with highlight placeholders inserted based on evidence.

    This is the main API used by the renderer. After further processing (markdown),
    callers should post-process HTML to replace placeholders with <mark> tags.
    """
    spans = compute_highlight_spans(text, evidence, n=n, overlap_threshold=overlap_threshold)
    if not spans:
        return text
    return insert_highlight_placeholders(text, spans) 