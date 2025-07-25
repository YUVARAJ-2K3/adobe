

import fitz  # PyMuPDF
import os
import json
import re
from collections import Counter

# --- Configuration ---
INPUT_DIR = "input"
OUTPUT_DIR = "output"

def process_pdf_to_outline(pdf_path):
    """
    Extracts a structured outline using a robust, multi-stage analysis.
    This version uses a "dominant style" method to handle emphasized text
    within paragraphs correctly.
    """
    doc = fitz.open(pdf_path)
    lines_data = []
    style_counts = Counter()

    extraction_flags = fitz.TEXT_PRESERVE_LIGATURES
    for page_num, page in enumerate(doc):
        page_dict = page.get_text("dict", flags=extraction_flags)
        for block in page_dict.get("blocks", []):
            if block['type'] == 0:
                for line in block.get("lines", []):
                    if not line.get("spans"):
                        continue
                    
                    # --- NEW: Dominant Style Analysis ---
                    # Instead of using just the first span, we analyze all spans in the line
                    # to find the most common style, preventing single bold/italic words
                    # from misclassifying a whole paragraph.
                    span_styles = Counter()
                    for s in line['spans']:
                        style_tuple = (
                            round(s['size']),
                            (s['flags'] & 2**1) != 0,  # is_bold
                            s['color']
                        )
                        # Weight the style by the number of characters to be more accurate
                        span_styles[style_tuple] += len(s['text'])
                    
                    # The most frequent style in the line is its dominant style
                    if not span_styles: continue
                    dominant_style = span_styles.most_common(1)[0][0]
                    
                    line_text = " ".join(s['text'] for s in line.get("spans", [])).strip()
                    if not line_text: continue

                    lines_data.append({
                        "text": line_text, "style": dominant_style, "font_size": dominant_style[0],
                        "page": page_num + 1, "bbox": line['bbox']
                    })
                    style_counts[dominant_style] += 1
    doc.close()

    if not lines_data:
        return {"title": "Empty Document", "outline": []}

    # Step 2: Determine the body style (most frequent dominant style)
    body_style = style_counts.most_common(1)[0][0] if style_counts else None

    # Step 3: Identify the Title
    page1_lines = sorted([l for l in lines_data if l['page'] == 1], key=lambda x: x['font_size'], reverse=True)
    title = page1_lines[0]['text'] if page1_lines else "Untitled Document"

    # Step 4: Gather potential heading lines
    all_candidates = [l for l in lines_data if l['style'] != body_style and l['text'] != title]

    # Step 5: Intelligently determine the primary H1 style
    numbered_h1_candidates = [
        l for l in all_candidates if re.match(r'^\d+\.[\sA-Z]', l['text']) and not re.match(r'^\d+\.\d+', l['text'])
    ]
    if numbered_h1_candidates:
        h1_style = Counter(l['style'] for l in numbered_h1_candidates).most_common(1)[0][0]
    else:
        candidate_styles = Counter(l['style'] for l in all_candidates)
        h1_style = candidate_styles.most_common(1)[0][0] if candidate_styles else None

    if not h1_style:
        return {"title": title, "outline": []}
    h1_font_size = h1_style[0]

    # Step 6: Group valid candidates by their style
    heading_groups = {}
    for line in all_candidates:
        if len(line['text'].split()) < 40 and not line['text'].endswith(('.', ':')):
            style = line['style']
            if style not in heading_groups: heading_groups[style] = []
            heading_groups[style].append(line)

    # Step 7: Rank styles relative to the H1 style
    level_map = {}
    for s in heading_groups:
        if s[0] >= h1_font_size:
            level_map[s] = "H1"
            
    subheading_styles = sorted([s for s in heading_groups if s[0] < h1_font_size], key=lambda s: s[0], reverse=True)
    if len(subheading_styles) > 0:
        level_map[subheading_styles[0]] = "H2"
    if len(subheading_styles) > 1:
        level_map[subheading_styles[1]] = "H3"
        
    # Step 8: Build the final outline
    outline = []
    for style, lines in heading_groups.items():
        if style in level_map:
            level = level_map[style]
            for line_info in lines:
                outline.append({"level": level, **line_info})

    outline.sort(key=lambda x: (x["page"], x["bbox"][1]))
    final_outline = [{"level": o['level'], "text": o['text'], "page": o['page']} for o in outline]

    return {"title": title, "outline": final_outline}


# --- Main Execution Block ---
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    print("Starting PDF processing...")
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(".pdf"):
            print(f"Processing: {filename}")
            input_path = os.path.join(INPUT_DIR, filename)
            output_filename = filename.rsplit('.', 1)[0] + ".json"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            try:
                result_json = process_pdf_to_outline(input_path)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result_json, f, ensure_ascii=False, indent=2)
                print(f"Successfully created: {output_filename}")
            except Exception as e:
                print(f"An error occurred while processing {filename}: {e}")
    print("\nProcessing complete.")


