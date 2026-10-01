import os
import re
import shutil
from pathlib import Path
from bs4 import BeautifulSoup, NavigableString, Tag

ROOT = Path(__file__).resolve().parents[1]
HTML_FILE = ROOT / "estudio_etapas.html"
MD_FILE = ROOT / "estudio_etapas.md"
IMG_DIR = ROOT / "imagenes"
IMG_DIR.mkdir(parents=True, exist_ok=True)

def inline_to_md(tag):
    if isinstance(tag, NavigableString):
        return str(tag)
    if not isinstance(tag, Tag):
        return ""
    
    name = tag.name
    inner = "".join(inline_to_md(c) for c in tag.children)
    
    if name in ("strong", "b"):
        return f"**{inner.strip()}**" if inner.strip() else ""
    elif name in ("em", "i"):
        return f"*{inner.strip()}*" if inner.strip() else ""
    elif name == "code":
        return f"`{inner.strip()}`" if inner.strip() else ""
    elif name == "a":
        href = tag.get("href", "")
        return f"[{inner.strip()}]({href})" if inner.strip() else ""
    elif name in ("span", "small"):
        return inner
    elif name == "br":
        return "\n"
    return inner

def table_to_md(table):
    rows = []
    for tr in table.find_all("tr"):
        cells = []
        for c in tr.find_all(["th", "td"]):
            # inline formatting inside table cells
            txt = " ".join("".join(inline_to_md(child) for child in c.children).split()).replace("|", "\\|")
            cells.append(txt)
        if cells:
            rows.append(cells)
    if not rows:
        return ""
    header = rows[0]
    sep = [":---" for _ in header]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    for r in rows[1:]:
        if len(r) < len(header):
            r = r + [""] * (len(header) - len(r))
        elif len(r) > len(header):
            r = r[:len(header)]
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)

def process_prose_child(child):
    if isinstance(child, NavigableString):
        s = str(child).strip()
        return [s] if s else []
    if not isinstance(child, Tag):
        return []
    
    lines = []
    name = child.name
    
    if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
        level = int(name[1])
        prefix = "#" * max(level, 3)
        lines.append(f"{prefix} {''.join(inline_to_md(c) for c in child.children).strip()}\n")
    elif name == "p":
        txt = "".join(inline_to_md(c) for c in child.children).strip()
        if txt:
            lines.append(f"{txt}\n")
    elif name == "ul":
        for li in child.find_all("li", recursive=False):
            txt = "".join(inline_to_md(c) for c in li.children).strip()
            # replace internal newlines with space or indent
            txt = " ".join(txt.split())
            if txt:
                lines.append(f"- {txt}")
        lines.append("")
    elif name == "ol":
        for idx, li in enumerate(child.find_all("li", recursive=False), 1):
            txt = "".join(inline_to_md(c) for c in li.children).strip()
            txt = " ".join(txt.split())
            if txt:
                lines.append(f"{idx}. {txt}")
        lines.append("")
    elif name == "pre":
        code_txt = child.get_text()
        lines.append(f"```text\n{code_txt.strip()}\n```\n")
    elif name == "table":
        lines.append(table_to_md(child) + "\n")
    elif name == "div" and "table-wrap" in child.get("class", []):
        t = child.find("table")
        if t:
            lines.append(table_to_md(t) + "\n")
    elif name == "div":
        for sub in child.children:
            lines.extend(process_prose_child(sub))
    return lines

def format_fig(figure):
    img = figure.find("img")
    if not img or not img.get("src"):
        return []
    src = img["src"]
    caption_tag = figure.find("figcaption")
    caption = caption_tag.get_text(strip=True) if caption_tag else (img.get("alt") or Path(src).stem)
    
    # Copy image to imagenes/
    clean_name = src.replace("figures/", "").replace("/", "_")
    src_file = ROOT / src.replace("/", os.sep)
    dst_file = IMG_DIR / clean_name
    if src_file.exists():
        shutil.copy2(src_file, dst_file)
        
    out = [f"![{caption}](imagenes/{clean_name})\n"]
    
    # Insert text block generated in step 2-3
    md_file = src_file.with_suffix(".md")
    if md_file.exists():
        content = md_file.read_text(encoding="utf-8").strip()
        out.append(f"\n{content}\n")
    else:
        out.append(f"\n> [transcrito de imagen, verificar]: {caption}\n")
    return out

def build_markdown():
    with open(HTML_FILE, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
        
    md = []
    
    # 1. HERO
    hero = soup.find("header", class_="hero")
    if hero:
        eyebrow = hero.find("p", class_="eyebrow")
        if eyebrow:
            md.append(f"> **{eyebrow.get_text(strip=True)}**\n")
        h1 = hero.find("h1")
        if h1:
            md.append(f"# {h1.get_text(strip=True)}\n")
        lede = hero.find("p", class_="lede")
        if lede:
            md.append(f"{''.join(inline_to_md(c) for c in lede.children).strip()}\n")
            
        kpis = hero.find_all("div", class_="kpi")
        if kpis:
            md.append("### Resumen de Métricas Principales (KPIs)\n")
            kpi_rows = [
                "| Indicador | Valor | Descripción |",
                "| :--- | :--- | :--- |"
            ]
            for kpi in kpis:
                k = kpi.find("span", class_="k").get_text(strip=True)
                v = kpi.find("span", class_="v").get_text(strip=True)
                s = kpi.find("span", class_="s").get_text(strip=True)
                kpi_rows.append(f"| **{k}** | `{v}` | {s} |")
            md.append("\n".join(kpi_rows) + "\n\n---\n")

    # 2. SECTIONS
    sections = soup.find_all("section", class_="sec")
    for sec in sections:
        sec_head = sec.find("header", class_="sec-head")
        if sec_head:
            eyebrow = sec_head.find("p", class_="eyebrow")
            if eyebrow:
                md.append(f"\n> **{eyebrow.get_text(strip=True).upper()}**\n")
            h2 = sec_head.find("h2")
            if h2:
                md.append(f"## {h2.get_text(strip=True)}\n")
                
        intro = sec.find("div", class_="intro")
        if intro:
            for c in intro.children:
                md.extend(process_prose_child(c))
            md.append("\n---\n")
            
        # STEPS
        steps = sec.find_all("article", class_="step")
        for step in steps:
            head = step.find("header", class_="step-head")
            if head:
                step_id_tag = head.find("span", class_="step-id")
                h3 = head.find("h3")
                codelink = head.find("a", class_="codelink")
                tags = [t.get_text(strip=True) for t in head.find_all("span", class_="tag")]
                tag_str = f" `[{', '.join(tags)}]`" if tags else ""
                
                step_id = step_id_tag.get_text(strip=True) if step_id_tag else ""
                s_title = h3.get_text(strip=True) if h3 else ""
                
                if step_id:
                    md.append(f"\n### Paso {step_id}: {s_title}{tag_str}\n")
                else:
                    md.append(f"\n### {s_title}{tag_str}\n")
                    
                if codelink and codelink.get("href"):
                    c_text = " ".join(codelink.get_text(separator=" ").split())
                    md.append(f"> *{c_text}* — [{codelink['href']}]({codelink['href']})\n")
                
            body = step.find("div", class_="step-body")
            if not body:
                continue
                
            for child in body.children:
                if not isinstance(child, Tag):
                    continue
                classes = child.get("class", [])
                
                # Standalone prose
                if "prose" in classes:
                    for p_child in child.children:
                        md.extend(process_prose_child(p_child))
                        
                # Figures
                elif "figs" in classes:
                    for figure in child.find_all("figure", class_="fig"):
                        md.extend(format_fig(figure))
                        
                # Collapsible table figures (details.tabfigs)
                elif "tabfigs" in classes or child.name == "details":
                    summary = child.find("summary")
                    if summary:
                        count_tag = summary.find("span", class_="count")
                        count_val = count_tag.get_text(strip=True) if count_tag else ""
                        sum_base = summary.find(text=True, recursive=False)
                        sum_text = str(sum_base).strip() if sum_base else "Tablas de este paso como imagen"
                        if count_val:
                            sum_text += f" ({count_val})"
                    else:
                        sum_text = "Tablas de este paso como imagen"
                    md.append(f"\n#### {sum_text}\n")
                    for figure in child.find_all("figure", class_="fig"):
                        md.extend(format_fig(figure))

    content = "\n".join(md)
    # clean multiple blank lines
    content = re.sub(r"\n{3,}", "\n\n", content)
    
    with open(MD_FILE, "w", encoding="utf-8") as f:
        f.write(content)
        
    print(f"Generated {MD_FILE} successfully ({len(content)} characters, {len(content.splitlines())} lines).")

if __name__ == "__main__":
    build_markdown()
