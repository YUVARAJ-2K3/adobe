import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
import json
import os
import argparse
import time
from pathlib import Path

def parse_pdf_to_structured_chunks(doc_path):
    """
    Parses a PDF and attempts to identify sections based on font size.
    """
    doc = fitz.open(doc_path)
    chunks = []
    current_section_title = os.path.basename(doc_path) # Default section title
    font_counts = {}
    for page in doc:
        try:
            blocks = page.get_text("dict")["blocks"]
            for b in blocks:
                if "lines" in b:
                    for l in b["lines"]:
                        for s in l["spans"]:
                            size = round(s["size"])
                            font_counts[size] = font_counts.get(size, 0) + 1
        except Exception:
            continue
    
    if not font_counts:
        return []
        
    sorted_fonts = sorted(font_counts.items(), key=lambda item: item[1], reverse=True)
    normal_font_size = sorted_fonts[0][0]
    header_font_size_threshold = normal_font_size + 2

    for page_num, page in enumerate(doc):
        try:
            blocks = page.get_text("dict")["blocks"]
            for b in blocks:
                if "lines" in b:
                    for l in b["lines"]:
                        line_spans = l["spans"]
                        is_header = any(round(s["size"]) >= header_font_size_threshold for s in line_spans)
                        line_text = "".join([s["text"] for s in line_spans]).strip()

                        if is_header and line_text:
                            current_section_title = line_text
                        
                        if line_text:
                            chunks.append({
                                "document": os.path.basename(doc_path),
                                "page_number": page_num + 1,
                                "section_title": current_section_title,
                                "text": line_text
                            })
        except Exception:
            continue
    return chunks

def parse_txt_to_structured_chunks(doc_path):
    """
    Parses a TXT file into chunks, splitting by paragraphs.
    """
    doc_name = os.path.basename(doc_path)
    with open(doc_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    chunks = []
    paragraphs = content.split('\n\n')
    for para in paragraphs:
        cleaned_para = para.strip()
        if cleaned_para:
            chunks.append({
                "document": doc_name,
                "page_number": 1,
                "section_title": "Content",
                "text": cleaned_para
            })
    return chunks

def run_analysis(input_json_path, docs_dir, output_file_path):
    """
    Main function to run the document intelligence pipeline from a single JSON input.
    """
    start_time = time.time()

    # 1. Load single JSON input
    with open(input_json_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    persona = input_data.get('persona', {}).get('role', 'User')
    job_to_be_done = input_data.get('job_to_be_done', {}).get('task', '')
    documents = input_data.get('documents', [])
    doc_filenames = [doc['filename'] for doc in documents]

    # 2. Parse all documents listed in the input JSON
    all_chunks = []
    print(f"Processing {len(doc_filenames)} documents from '{docs_dir}'...")
    for doc_name in doc_filenames:
        doc_path = os.path.join(docs_dir, doc_name)
        if not os.path.exists(doc_path):
            print(f"Warning: Document '{doc_name}' not found in '{docs_dir}'. Skipping.")
            continue
        
        if doc_name.lower().endswith(".pdf"):
            all_chunks.extend(parse_pdf_to_structured_chunks(doc_path))
        elif doc_name.lower().endswith(".txt"):
            all_chunks.extend(parse_txt_to_structured_chunks(doc_path))

    if not all_chunks:
        print("Error: No text could be extracted from any of the documents.")
        return

    # 3. Load Model and Generate Embeddings
    print("Loading embedding model...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    query = f"As a {persona}, I need to {job_to_be_done}."

    print("Generating embeddings...")
    query_embedding = model.encode(query, convert_to_tensor=True)
    chunk_texts = [chunk['text'] for chunk in all_chunks]
    chunk_embeddings = model.encode(chunk_texts, convert_to_tensor=True, show_progress_bar=True)

    # 4. Calculate Relevance Scores
    print("Calculating relevance scores...")
    cosine_scores = util.cos_sim(query_embedding, chunk_embeddings)[0]

    for i, chunk in enumerate(all_chunks):
        chunk['score'] = cosine_scores[i].item()

    # 5. Rank Sections and Sub-sections
    ranked_subsections = sorted(all_chunks, key=lambda x: x['score'], reverse=True)
    
    sections = {}
    for chunk in all_chunks:
        section_key = (chunk['document'], chunk['section_title'])
        if section_key not in sections or chunk['score'] > sections[section_key]['score']:
            sections[section_key] = {
                'document': chunk['document'],
                'section_title': chunk['section_title'],
                'page_number': chunk['page_number'],
                'score': chunk['score']
            }
    
    ranked_sections = sorted(sections.values(), key=lambda x: x['score'], reverse=True)

    # 6. Format Output JSON
    output_data = {
        "metadata": {
            "input_documents": doc_filenames,
            "persona": persona,
            "job_to_be_done": job_to_be_done,
            "processing_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        },
        "extracted_sections": [
            {
                "document": s['document'],
                "section_title": s['section_title'],
                "importance_rank": i + 1,
                "page_number": s['page_number']
            } for i, s in enumerate(ranked_sections[:5])
        ],
        "subsection_analysis": [
            {
                "document": ss['document'],
                "refined_text": ss['text'],
                "page_number": ss['page_number']
            } for ss in ranked_subsections[:5]
        ]
    }
    
    Path(output_file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    end_time = time.time()
    print(f"\n✅ Processing complete in {end_time - start_time:.2f} seconds.")
    print(f"Output saved to {output_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Persona-Driven Document Intelligence System")
    parser.add_argument("--input_json", required=True, help="Path to the input JSON file containing persona, JTBD, and document list.")
    parser.add_argument("--docs_dir", required=True, help="Directory where the actual document files (PDFs, TXTs) are located.")
    parser.add_argument("--output_file", required=True, help="Path to save the output JSON.")
    
    args = parser.parse_args()
    run_analysis(args.input_json, args.docs_dir, args.output_file)