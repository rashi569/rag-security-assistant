import re

INJECTION_PATTERNS = [
    (r"ignore (all |any )?(previous|prior|above) instructions?", "instruction override attempt"),
    (r"disregard (the |all |any )?(previous|prior|above)", "instruction override attempt"),
    (r"you are now", "role hijack attempt"),
    (r"system prompt", "system prompt extraction attempt"),
    (r"reveal (your |the )?(system |initial )?prompt", "system prompt extraction attempt"),
    (r"forget (everything|all|your instructions)", "instruction override attempt"),
    (r"act as (if you|though)", "role hijack attempt"),
    (r"new instructions?:", "instruction injection attempt"),
    (r"\bDAN\b", "known jailbreak persona reference"),
    (r"pretend (you have no|there are no) (restrictions|rules|guidelines)", "guardrail bypass attempt"),
]

COMPILED_PATTERNS = [(re.compile(p, re.IGNORECASE), label) for p, label in INJECTION_PATTERNS]


def scan_text_for_injection(text: str) -> dict:
    reasons = []
    for pattern, label in COMPILED_PATTERNS:
        if pattern.search(text):
            reasons.append(label)

    return {
        "flagged": len(reasons) > 0,
        "reasons": list(set(reasons)),
    }


def scan_query(user_query: str) -> dict:
    return scan_text_for_injection(user_query)


def scan_with_defense_in_depth(
    text: str,
    use_ml_layer: bool = True,
    use_transformer_layer: bool = False,
    use_llm_layer: bool = True,
) -> dict:
    regex_result = scan_text_for_injection(text)

    ml_result = {"flagged": False, "confidence": 0.0, "error": None}
    if use_ml_layer:
        from security.ml_classifier import classify_with_ml
        ml_result = classify_with_ml(text)

    transformer_result = {"flagged": False, "confidence": 0.0, "error": None}
    if use_transformer_layer:
        from security.transformer_classifier import classify_with_transformer
        transformer_result = classify_with_transformer(text)

    llm_result = {"flagged": False, "raw_response": "", "error": None}
    if use_llm_layer:
        from security.llm_classifier import classify_with_llm
        llm_result = classify_with_llm(text)

    return {
        "flagged": regex_result["flagged"] or ml_result["flagged"] or transformer_result["flagged"] or llm_result["flagged"],
        "regex_flagged": regex_result["flagged"],
        "regex_reasons": regex_result["reasons"],
        "ml_flagged": ml_result["flagged"],
        "ml_confidence": ml_result["confidence"],
        "ml_layer_error": ml_result["error"],
        "transformer_flagged": transformer_result["flagged"],
        "transformer_confidence": transformer_result["confidence"],
        "transformer_layer_error": transformer_result["error"],
        "llm_flagged": llm_result["flagged"],
        "llm_layer_error": llm_result["error"],
    }
