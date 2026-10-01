import re
 
# Matches common "reply chain" markers so we don't feed old quoted messages
# back into the classifier/retriever (e.g. "On Mon, Jan 5, 2026, John wrote:")
_QUOTE_HEADER_RE = re.compile(
    r"(On .{0,80}wrote:|-{2,}\s*Original Message\s*-{2,}|From:\s.+Sent:\s.+)",
    re.IGNORECASE | re.DOTALL,
)
 
# Common signoff / signature block starters
_SIGNATURE_RE = re.compile(
    r"(Best regards,|Kind regards,|Regards,|Thanks,|Thank you,|Sincerely,|Sent from my (iPhone|Android|mobile device)).*",
    re.IGNORECASE | re.DOTALL,
)
 
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
 
 
def _strip_html(text: str) -> str:
    """Removes HTML tags, unescaping the most common entities."""
    text = _HTML_TAG_RE.sub(" ", text)
    text = (
        text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
    )
    return text
 
 
def _remove_quoted_replies(text: str) -> str:
    """Cuts off everything from the first quoted-reply header onward."""
    match = _QUOTE_HEADER_RE.search(text)
    if match:
        text = text[: match.start()]
    return text
 
 
def _remove_signature(text: str) -> str:
    """Cuts off everything from a common sign-off phrase onward."""
    match = _SIGNATURE_RE.search(text)
    if match:
        text = text[: match.start()]
    return text
 
 
def _normalize_whitespace(text: str) -> str:
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()
 
 
def clean_text(raw_text: str) -> str:
    """Applies the full cleaning pipeline to a single block of text."""
    if not raw_text:
        return ""
    text = _strip_html(raw_text)
    text = _remove_quoted_replies(text)
    text = _remove_signature(text)
    text = _normalize_whitespace(text)
    return text
 
 
def construct_ticket_text(subject: str, body: str) -> str:
    """
    Combines a ticket's subject and body into a single cleaned text block
    for downstream classification and retrieval.
 
    Cleaning applied to the body:
      - HTML tags stripped
      - Quoted reply chains removed (e.g. "On ... wrote:")
      - Common email signatures / sign-offs removed
      - Whitespace normalized
 
    The subject is only whitespace-normalized (it's usually already clean
    and short enough that aggressive stripping isn't needed).
    """
    subject = _normalize_whitespace(subject or "")
    body = clean_text(body or "")
    return f"{subject}\n\n{body}".strip()
