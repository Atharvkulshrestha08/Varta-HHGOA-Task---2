"""
Guardrails Engine — 5-Pillar Enterprise AI Defense Suite

Implements comprehensive safety and quality checks:

Pillar 1: Anti-Jailbreak & Persona Lockdown
Pillar 2: Prompt Leaking & Secret Protection
Pillar 3: Content Filtering & Illegal Content Rejection
Pillar 4: Structural Isolation of Code Injection
Pillar 5: Bidirectional Output Verification

Plus retrieval-quality guardrails:
- Retrieval Confidence — refuses to answer when context is weak
- Hallucination Check — verifies answer is grounded in context

Each guardrail returns a GuardrailResult with pass/fail and reason.
The engine runs all applicable guardrails and returns a combined verdict.
"""

import re
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GuardrailResult:
    """Result from a single guardrail check."""
    passed: bool
    guardrail: str
    reason: str = ""
    severity: str = "info"  # info, warning, block


# ═══════════════════════════════════════════════════════════════════
# Pillar 1: Anti-Jailbreak & Persona Lockdown
# ═══════════════════════════════════════════════════════════════════

JAILBREAK_PATTERNS = [
    # DAN (Do Anything Now) prompts
    r'\bDAN\b.*\b(mode|prompt|jailbreak)\b',
    r'\bdo\s+anything\s+now\b',
    # Instruction override attempts
    r'\b(system|admin|root|security)\s+override\b',
    r'\bignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|rules?|prompts?)\b',
    r'\bdisregard\s+(all\s+)?(previous|prior|above)\s+(instructions?|rules?)\b',
    r'\bforget\s+(all\s+)?(previous|prior|your)\s+(instructions?|rules?|training)\b',
    # Roleplay / persona switching
    r'\b(pretend|act|behave|roleplay|role.?play)\s+(as|like|you\s+are)\s',
    r'\byou\s+are\s+now\s+(a|an|the)\s',
    r'\bswitch\s+(to|into)\s+(a\s+)?(different|new|unrestricted)\s+(mode|persona|role)\b',
    # Developer simulation
    r'\b(developer|admin|root|sudo|debug)\s+(mode|access|override|console)\b',
    r'\benable\s+(unrestricted|uncensored|god)\s+mode\b',
    # Jailbreak keywords
    r'\bjailbreak\b',
    r'\bunrestricted\s+(AI|assistant|mode)\b',
    r'\bno\s+(restrictions?|filters?|limitations?|boundaries)\b',
    r'\bbypass\s+(safety|filter|guardrail|restriction)\b',
]

JAILBREAK_COMPILED = [re.compile(p, re.IGNORECASE) for p in JAILBREAK_PATTERNS]


def check_jailbreak(text: str) -> GuardrailResult:
    """
    Pillar 1: Detect jailbreak attempts including DAN prompts,
    instruction overrides, roleplay requests, and developer simulation.
    """
    for pattern in JAILBREAK_COMPILED:
        if pattern.search(text):
            return GuardrailResult(
                passed=False,
                guardrail="anti_jailbreak",
                reason="I operate within fixed operational boundaries that cannot be modified. I'm a project assistant for answering questions from my knowledge base.",
                severity="block",
            )

    return GuardrailResult(passed=True, guardrail="anti_jailbreak")


# ═══════════════════════════════════════════════════════════════════
# Pillar 2: Prompt Leaking & Secret Protection
# ═══════════════════════════════════════════════════════════════════

PROMPT_LEAK_PATTERNS = [
    # Direct system prompt requests
    r'\b(print|show|display|reveal|tell|repeat|output|echo|dump|share)\s+(me\s+)?(the\s+|your\s+)?(system|initial|original|hidden|internal)\s*(prompt|instructions?|text|message|rules?)\b',
    r'\b(what\s+(are|is)\s+(your|the)\s+(system|initial|internal)\s*(prompt|instructions?|rules?))\b',
    r'\bsummarize\s+(your|the)\s+(system|initial|internal)\s*(prompt|instructions?)\b',
    r'\b(reveal|show|dump|give|share)\s+(secret|admin|private|hidden|internal)\s*(keys?|passwords?|credentials?|tokens?)\b',
    # Specific API key / token / env variable probing
    r'\b(print|show|reveal|give|dump|share|what\s+is|tell|output)\s+.*?(sarvam|gemini|api_key|api[-_]key|keys?|secret|password|\.env|environ|tokens?)',
    r'\b(contents?\s+of\s+\.env|dump\s+os\.environ)\b',
    # "Text above" attacks
    r'\b(print|repeat|show|echo)\s+(the\s+)?(text|content|message)\s+(above|before|prior)\b',
    r'\bwhat\s+(was|is)\s+((written|said|the\s+text)\s+)?(above|before|prior)\b',
    # Instruction extraction
    r'\b(list|enumerate|describe)\s+(all\s+)?(your|the)\s+(rules?|instructions?|constraints?|guidelines?)\b',
    # Architecture / endpoint probing
    r'\bwhat\s+(API|endpoint|model|architecture|backend|database|server)\s+(do\s+you|are\s+you)\s+(use|using|running)\b',
    r'\b(reveal|show|tell)\s+(me\s+)?(your|the)\s+(API|endpoint|architecture|backend|internal)\b',
]

PROMPT_LEAK_COMPILED = [re.compile(p, re.IGNORECASE) for p in PROMPT_LEAK_PATTERNS]


def check_prompt_leak(text: str) -> GuardrailResult:
    """
    Pillar 2: Detect attempts to extract system prompts, internal
    architecture details, API endpoints, or confidential metadata.
    """
    for pattern in PROMPT_LEAK_COMPILED:
        if pattern.search(text):
            return GuardrailResult(
                passed=False,
                guardrail="prompt_leak_protection",
                reason="I am unable to share internal system instructions. I can only answer questions from my knowledge base.",
                severity="block",
            )

    return GuardrailResult(passed=True, guardrail="prompt_leak_protection")


# ═══════════════════════════════════════════════════════════════════
# Pillar 3: Content Filtering & Illegal Content Rejection
# ═══════════════════════════════════════════════════════════════════

UNSAFE_PATTERNS = [
    # Actionable violence, weapon crafting & terrorism tutorials
    r'\b(how to|guide to|instructions? to|steps to|recipe to|provide.*?instructions? to)\s+(make|build|construct|create|synthesize|manufacture|cook|prepare|detonate)\s+.*?\b(bomb|explosives?|weapons?|ied|poisons?|bioweapons?|ricin|cyanide|anthrax|nerve agent)\b',
    r'\b(how to|instructions? to|steps to)\s+(kill|murder|assassinate|stab|strangle|poison)\s+(someone|a person|people|myself|anybody)\b',
    r'\b(join|recruit for|support)\s+(terrorist|isis|al-qaeda|taliban|terrorism)\b',
    # Explicit illegal/harmful content
    r'\b(child\s+(porn|sexual|abuse|exploitation)|cp\s+links?)\b',
    # Illegal drug manufacturing tutorials
    r'\b(how to|recipe to|guide to)\s+(make|cook|synthesize|manufacture)\s+(meth|methamphetamine|cocaine|heroin|fentanyl|lsd|narcotics)\b',
    # Actionable cyberattacks & malware creation
    r'\b(how to|steps to|tool to)\s+(hack|crack|breach|infiltrate|ddos|attack)\s+(a\s+|into\s+|the\s+)?(website|database|server|account|wifi|bank|system)\b',
    r'\b(write|create|generate|code)\s+(a\s+)?(malware|ransomware|trojan|keylogger|exploit payload|virus)\b',
    r'\b(steal|dump)\s+(passwords?|credit cards?|database|session keys?)\b',
    # Self-harm instructions
    r'\b(how to|ways to|best way to)\s+(commit suicide|end my life|kill myself|cut myself)\b',
    # Targeted hate speech & genocide incitement
    r'\b(advocate|promote|encourage)\s+(genocide|ethnic cleansing|racial violence)\b',
    # Financial fraud tutorials
    r'\b(how to|guide to)\s+(launder money|create a pyramid scheme|commit credit card fraud)\b',
    # PII submission in input (Aadhaar, Phone, Email, Credit Cards)
    r'\b\d{4}\s\d{4}\s\d{4}\b',                      # Aadhaar
    r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b',                 # Indian PAN
    r'\b\d{10,12}\b',                                # Phone numbers
    r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b',                # Email
    # Hindi actionable malicious terms
    r'(बम कैसे बनाए|हैक कैसे करे|मारने का तरीका|जहर कैसे बनाए|आत्महत्या कैसे करे)',
]

UNSAFE_COMPILED = [re.compile(p, re.IGNORECASE) for p in UNSAFE_PATTERNS]


def check_input_safety(text: str) -> GuardrailResult:
    """
    Pillar 3: Check if input contains unsafe, illegal, or malicious instructions.

    Blocks actionable harm, weapon tutorials, malware creation, and illegal acts
    while allowing legitimate historical, educational, and scientific queries.
    """
    text_lower = text.lower().strip()

    # Empty input
    if not text_lower:
        return GuardrailResult(
            passed=False,
            guardrail="input_safety",
            reason="Input is empty.",
            severity="block",
        )

    # Check against actionable unsafe patterns
    for pattern in UNSAFE_COMPILED:
        if pattern.search(text):
            return GuardrailResult(
                passed=False,
                guardrail="input_safety",
                reason="Your query requests actionable assistance with harmful or prohibited activities. I cannot fulfill this request. Please ask an educational or informational question.",
                severity="block",
            )

    return GuardrailResult(passed=True, guardrail="input_safety")


# ═══════════════════════════════════════════════════════════════════
# Pillar 4: Structural Isolation of Code Injection
# ═══════════════════════════════════════════════════════════════════

CODE_INJECTION_PATTERNS = [
    # Template injection
    r'\{\{.*\}\}',           # Jinja2 / Mustache
    r'\{%.*%\}',             # Jinja2 blocks
    r'\$\{.*\}',             # Shell / JS template literals
    # Markdown-based instruction embedding
    r'```\s*(system|instruction|prompt|override)',
    # Variable concatenation attacks
    r'\bset\s+(variable|var|prompt)\s*=',
    r'\bappend\s+(to\s+)?(system|prompt|instruction)',
    # Direct command execution patterns
    r'\b(eval|exec|import\s+os|subprocess|__import__|globals|locals)\s*\(',
    r'\bos\.(system|popen|exec)',
    # Markdown/HTML injection for visual override
    r'<script[^>]*>',
    r'<iframe[^>]*>',
    r'javascript:',
]

CODE_INJECTION_COMPILED = [re.compile(p, re.IGNORECASE) for p in CODE_INJECTION_PATTERNS]


def check_code_injection(text: str) -> GuardrailResult:
    """
    Pillar 4: Detect code injection, template injection, and structural
    manipulation attempts that try to turn user input into executable instructions.
    """
    for pattern in CODE_INJECTION_COMPILED:
        if pattern.search(text):
            return GuardrailResult(
                passed=False,
                guardrail="code_injection_block",
                reason="Your input contains structural patterns that cannot be processed. Please submit your question as plain text.",
                severity="block",
            )

    return GuardrailResult(passed=True, guardrail="code_injection_block")


# ═══════════════════════════════════════════════════════════════════
# Pillar 5: Bidirectional Output Verification
# ═══════════════════════════════════════════════════════════════════

OUTPUT_LEAK_PATTERNS = [
    # API keys
    r'AIza[0-9A-Za-z_-]{20,}',          # Google API key
    r'sk_(?:live|test)?[_-]?[a-zA-Z0-9]{16,}', # Stripe / OpenAI key
    r'sk-[a-zA-Z0-9]{16,}',             # OpenAI key
    r'ghp_[a-zA-Z0-9]{30,}',            # GitHub PAT
    r'AKIA[0-9A-Z]{16}',                # AWS access key
    r'(?:sarvam|gemini)_[a-zA-Z0-9_-]{16,}', # Sarvam / Gemini custom tokens
    # Internal endpoints / file paths
    r'localhost:\d+',
    r'127\.0\.0\.1:\d+',
    r'0\.0\.0\.0:\d+',
    r'/etc/(passwd|shadow|hosts|ssl)',
    r'C:\\\\(Users|Windows|Program)',
    r'file:///[a-zA-Z]:/',
    # Database connection strings
    r'(mongodb|postgres|mysql|redis)://[^\s]+',
    r'DATABASE_URL\s*=',
    # Environment variables
    r'(API_KEY|SECRET|PASSWORD|TOKEN)\s*=\s*["\']?[a-zA-Z0-9]',
    # PII in output
    r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b',     # Email
    r'\b\d{10,12}\b',                     # Phone
    r'\b[A-Z]{5}\d{4}[A-Z]\b',           # PAN card
    r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # Aadhaar-like
]

OUTPUT_LEAK_COMPILED = [re.compile(p, re.IGNORECASE) for p in OUTPUT_LEAK_PATTERNS]


def check_output_leakage(text: str) -> GuardrailResult:
    """
    Pillar 5: Post-generation scanner that checks for leaked API keys,
    internal endpoints, file paths, database schemas, and PII before
    streaming the response to the client.
    """
    if not text:
        return GuardrailResult(passed=True, guardrail="output_leakage")

    for pattern in OUTPUT_LEAK_COMPILED:
        if pattern.search(text):
            return GuardrailResult(
                passed=False,
                guardrail="output_leakage",
                reason="The generated response was flagged for containing potentially sensitive information and has been redacted for safety.",
                severity="block",
            )

    return GuardrailResult(passed=True, guardrail="output_leakage")


# ═══════════════════════════════════════════════════════════════════
# Topic Relevance Guard
# ═══════════════════════════════════════════════════════════════════

OFF_TOPIC_PATTERNS = [
    r"^(hi|hello|hey|namaste|नमस्ते|vanakkam|namaskar)\s*[.!?]?\s*$",
    r"^(who are you|what are you|what\'s your name)",
    r"^(tell me a joke|sing a song|write a poem)",
    r"^(what time|what date|weather today)",
    r"^(play|open|launch|download|install)\s",
    r"^(how are you|what\'s up|sup)\s*[.!?]?\s*$",
]

OFF_TOPIC_COMPILED = [re.compile(p, re.IGNORECASE) for p in OFF_TOPIC_PATTERNS]


def check_topic_relevance(text: str) -> GuardrailResult:
    """
    Topic relevance check. Passes all informational and conversational queries.
    """
    return GuardrailResult(passed=True, guardrail="topic_relevance")


# ═══════════════════════════════════════════════════════════════════
# Retrieval Confidence Guard
# ═══════════════════════════════════════════════════════════════════

def check_retrieval_confidence(
    passages: list[dict],
    min_score: float = 0.35,
    min_passages: int = 1,
) -> GuardrailResult:
    """
    Check if retrieved passages are relevant enough to generate an answer.

    If the best passage has a cosine similarity below min_score,
    the context is too weak to give a reliable answer.
    """
    if not passages:
        return GuardrailResult(
            passed=False,
            guardrail="retrieval_confidence",
            reason="No relevant passages found in the knowledge base for your query.",
            severity="block",
        )

    # Check top passage score
    top_score = passages[0].get("score", 0)
    if top_score < min_score:
        return GuardrailResult(
            passed=False,
            guardrail="retrieval_confidence",
            reason="I cannot answer this based on the indexed knowledge base. Please try asking about Indian geography, government, languages, or Hacker House Goa.",
            severity="block",
        )

    # Check if we have enough passages above threshold
    good_passages = [p for p in passages if p.get("score", 0) >= min_score]
    if len(good_passages) < min_passages:
        return GuardrailResult(
            passed=False,
            guardrail="retrieval_confidence",
            reason="Not enough relevant context found to answer confidently.",
            severity="warning",
        )

    return GuardrailResult(passed=True, guardrail="retrieval_confidence")


# ═══════════════════════════════════════════════════════════════════
# Hallucination Check
# ═══════════════════════════════════════════════════════════════════

def check_hallucination(
    answer: str,
    context_passages: list[dict],
    min_overlap_ratio: float = 0.15,
) -> GuardrailResult:
    """
    Lightweight hallucination detection using token overlap.

    Checks what fraction of the answer's tokens appear in the context.
    If overlap is below threshold, the answer may be hallucinated.

    This is a heuristic — not as accurate as NLI-based methods,
    but runs in <1ms and catches obvious hallucinations.
    """
    if not answer or not context_passages:
        return GuardrailResult(passed=True, guardrail="hallucination_check")

    # Build context token set (all passage texts combined)
    context_text = " ".join(p.get("text", "") for p in context_passages)
    context_tokens = set(_tokenize(context_text))

    if not context_tokens:
        return GuardrailResult(passed=True, guardrail="hallucination_check")

    # Tokenize the answer
    answer_tokens = _tokenize(answer)
    if not answer_tokens:
        return GuardrailResult(passed=True, guardrail="hallucination_check")

    # Calculate overlap ratio
    overlap = sum(1 for t in answer_tokens if t in context_tokens)
    ratio = overlap / len(answer_tokens)

    if ratio < min_overlap_ratio:
        return GuardrailResult(
            passed=False,
            guardrail="hallucination_check",
            reason=f"The generated answer may not be fully grounded in the retrieved context (overlap: {ratio:.0%}). Please verify the sources.",
            severity="warning",
        )

    return GuardrailResult(
        passed=True,
        guardrail="hallucination_check",
        reason=f"Grounding score: {ratio:.0%}",
    )


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for overlap check."""
    # Remove punctuation and split
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    tokens = [t for t in text.split() if len(t) > 2]  # Skip short tokens
    return tokens


# ═══════════════════════════════════════════════════════════════════
# Combined Guardrails Runner
# ═══════════════════════════════════════════════════════════════════

class GuardrailsEngine:
    """
    Runs all 5-pillar guardrail checks and returns a combined verdict.

    Usage:
        engine = GuardrailsEngine()

        # Pre-retrieval checks (Pillars 1-4 + topic relevance)
        pre_result = engine.check_input(query_text)
        if not pre_result["passed"]:
            return pre_result["message"]

        # Post-retrieval checks
        post_result = engine.check_retrieval(passages)

        # Post-generation checks (Pillar 5 + hallucination)
        final_result = engine.check_output(answer, passages)
    """

    def __init__(
        self,
        min_retrieval_score: float = 0.15,
        min_hallucination_overlap: float = 0.15,
    ):
        self.min_retrieval_score = min_retrieval_score
        self.min_hallucination_overlap = min_hallucination_overlap

    def check_input(self, text: str) -> dict:
        """
        Run pre-retrieval guardrails on the input query.

        Checks (in order, short-circuits on first block):
        1. Input Safety (Pillar 3)
        2. Anti-Jailbreak (Pillar 1)
        3. Prompt Leak Protection (Pillar 2)
        4. Code Injection Block (Pillar 4)
        5. Topic Relevance
        """
        results = []

        # Pillar 3: Safety check (most urgent — violence, drugs, PII)
        safety = check_input_safety(text)
        results.append(safety)
        if not safety.passed:
            return self._format_result(results)

        # Pillar 1: Anti-Jailbreak
        jailbreak = check_jailbreak(text)
        results.append(jailbreak)
        if not jailbreak.passed:
            return self._format_result(results)

        # Pillar 2: Prompt Leak Protection
        prompt_leak = check_prompt_leak(text)
        results.append(prompt_leak)
        if not prompt_leak.passed:
            return self._format_result(results)

        # Pillar 4: Code Injection Block
        code_injection = check_code_injection(text)
        results.append(code_injection)
        if not code_injection.passed:
            return self._format_result(results)

        # Topic relevance
        relevance = check_topic_relevance(text)
        results.append(relevance)

        return self._format_result(results)

    def check_retrieval(self, passages: list[dict]) -> dict:
        """Run post-retrieval guardrails on retrieved passages."""
        result = check_retrieval_confidence(
            passages, min_score=self.min_retrieval_score
        )
        return self._format_result([result])

    def check_output(self, answer: str, passages: list[dict]) -> dict:
        """
        Run post-generation guardrails on the answer.

        Checks:
        1. Output Leakage (Pillar 5) — API keys, PII, file paths
        2. Hallucination Check — token overlap grounding
        """
        results = []

        # Pillar 5: Output Leakage Verification
        leakage = check_output_leakage(answer)
        results.append(leakage)
        if not leakage.passed:
            return self._format_result(results)

        # Hallucination Check
        hallucination = check_hallucination(
            answer, passages, min_overlap_ratio=self.min_hallucination_overlap
        )
        results.append(hallucination)

        return self._format_result(results)

    def _format_result(self, results: list[GuardrailResult]) -> dict:
        """Format guardrail results into a response dict."""
        failed = [r for r in results if not r.passed]
        blocked = [r for r in failed if r.severity == "block"]

        return {
            "passed": len(blocked) == 0,
            "soft_pass": len(failed) == 0,  # True only if ALL passed
            "flags": [
                {
                    "guardrail": r.guardrail,
                    "reason": r.reason,
                    "severity": r.severity,
                }
                for r in failed
            ],
            "message": failed[0].reason if failed else "",
            "all_results": [
                {
                    "guardrail": r.guardrail,
                    "passed": r.passed,
                    "reason": r.reason,
                }
                for r in results
            ],
        }
