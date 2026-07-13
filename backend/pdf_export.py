"""Export a session's full teaching-card history to a PDF file."""
import html
from io import BytesIO

from xhtml2pdf import pisa

_CSS = """
<style>
  body { font-family: Helvetica, Arial, sans-serif; font-size: 11px; color: #222; }
  h1 { font-size: 20px; border-bottom: 2px solid #333; padding-bottom: 6px; }
  h2.question { font-size: 15px; color: #1a1a1a; margin-top: 28px; background: #f0f0f0; padding: 6px; }
  h3 { font-size: 13px; color: #333; margin-bottom: 2px; margin-top: 14px; }
  .step { margin-bottom: 10px; }
  pre { background: #f5f5f5; padding: 8px; font-size: 10px; white-space: pre-wrap; }
  ul { margin-top: 2px; }
  .lbl { font-weight: bold; }
</style>
"""


def _esc(text) -> str:
    if text is None:
        return ""
    return html.escape(str(text))


def _list_html(items) -> str:
    if not items:
        return "<p><i>None</i></p>"
    return "<ul>" + "".join(f"<li>{_esc(i)}</li>" for i in items) + "</ul>"


def _line_by_line_html(items) -> str:
    if not items:
        return "<p><i>None</i></p>"
    rows = []
    for item in items:
        line = _esc(item.get("line", ""))
        expl = _esc(item.get("explanation", ""))
        rows.append(f"<div class='step'><span class='lbl'>{line}</span><br/>{expl}</div>")
    return "".join(rows)


def _sources_html(items) -> str:
    if not items:
        return "<p><i>General knowledge (no library documents matched this question)</i></p>"
    rows = []
    for item in items:
        source = _esc(item.get("source", ""))
        note = _esc(item.get("note", ""))
        rows.append(f"<li><b>{source}</b> — {note}</li>")
    return "<ul>" + "".join(rows) + "</ul>"


def markdown_to_html(md_text: str) -> str:
    import re
    if not md_text:
        return ""
    
    # Escape HTML tags first
    html_text = html.escape(md_text)
    
    # Replace triple backtick blocks with pre code
    parts = html_text.split("```")
    for i in range(1, len(parts), 2):
        code = parts[i]
        lines = code.split("\n")
        if lines and lines[0].strip() in ["python", "javascript", "js", "py", "bash", "html", "css", "sql", "cpp", "c++"]:
            lines.pop(0)
        clean_code = "\n".join(lines).strip()
        parts[i] = f"<pre><code>{clean_code}</code></pre>"
    html_text = "".join(parts)
    
    lines = html_text.split("\n")
    in_code = False
    for i, line in enumerate(lines):
        if line.startswith("<pre>"):
            in_code = True
        if line.endswith("</pre>") and in_code:
            in_code = False
            continue
        if in_code:
            continue
            
        # Headers
        if line.startswith("# "):
            lines[i] = f"<h1>{line[2:]}</h1>"
        elif line.startswith("## "):
            lines[i] = f"<h2>{line[3:]}</h2>"
        elif line.startswith("### "):
            lines[i] = f"<h3>{line[4:]}</h3>"
        elif line.startswith("- ") or line.startswith("* "):
            lines[i] = f"<li>{line[2:]}</li>"
        elif line.strip() != "":
            lines[i] = f"<p>{line}</p>"
            
        # Bold and inline code
        lines[i] = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", lines[i])
        lines[i] = re.sub(r"`(.*?)`", r"<code>\1</code>", lines[i])
        
    return "\n".join(lines)


def build_session_html(title: str, messages: list[dict]) -> str:
    parts = [f"<html><head>{_CSS}</head><body>", f"<h1>{_esc(title)}</h1>"]
    for i, msg in enumerate(messages, start=1):
        card = msg["card"]
        mode = msg.get("mode", "code")
        heading = "Problem" if mode == "leetcode" else "Topic Study Note"
        parts.append(f"<h2 class='question'>{heading} {i}: {_esc(msg['question'])}</h2>")
        
        md_text = card.get("text") or card.get("concept") or card.get("what_is_it") or ""
        if not md_text and isinstance(card, dict):
            # Fallback if card is old JSON structure: reconstruct a nice markdown string
            sub_parts = []
            for key in ["objective", "concept", "mental_model", "code", "exercise", "next_topic"]:
                if card.get(key):
                    sub_parts.append(f"## {key.title()}\n{card.get(key)}")
            md_text = "\n\n".join(sub_parts)
            
        parts.append(markdown_to_html(md_text))
        
        # Sources if any
        if card.get("sources"):
            parts.append(f"<h3>Sources</h3>{_sources_html(card.get('sources'))}")
            
    parts.append("</body></html>")
    return "".join(parts)


def export_session_pdf(title: str, messages: list[dict]) -> bytes:
    html_content = build_session_html(title, messages)
    buf = BytesIO()
    result = pisa.CreatePDF(src=html_content, dest=buf)
    if result.err:
        raise RuntimeError("Failed to generate PDF from session content.")
    return buf.getvalue()
