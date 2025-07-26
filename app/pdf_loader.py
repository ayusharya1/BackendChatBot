import re
from pathlib import Path
from PyPDF2 import PdfReader

def extract_metadata_from_folder(folder_path: Path):
    metadata_texts = []
    for pdf_path in folder_path.glob("*.pdf"):
        try:
            reader = PdfReader(str(pdf_path))  
            full_text = "\n".join([p.extract_text() or "" for p in reader.pages])  

            # 🔍 Try to extract abstract or summary
            abstract_match = re.search(
                r"(?:Abstract|Summary|ABSTRACT|SUMMARY)\s*[:\-]?\s*(.+?)(?=\n[A-Z][a-z]+|Chapter\s+\d+|Introduction|\Z)",
                full_text, re.IGNORECASE | re.DOTALL
            )  
            abstract = abstract_match.group(1).strip() if abstract_match else "Not available"  

            # 🔍 Basic metadata extraction  
            title = re.search(r"(Title|TITLE)\s*[:\-]?\s*(.*)", full_text, re.IGNORECASE)  
            author = re.search(r"(Author|Submitted by|Candidate)\s*[:\-]?\s*(.*)", full_text, re.IGNORECASE)  
            supervisor = re.search(r"(Supervisor|Guide|Advisor|Professor)\s*[:\-]?\s*(.*)", full_text, re.IGNORECASE)  
            year = re.search(r"\b(20[0-3][0-9])\b", full_text)  
            publisher = re.search(r"(Publisher)\s*[:\-]?\s*(.*)", full_text, re.IGNORECASE)  
            url = re.search(r"(http[s]?://\S+)", full_text)  

            metadata_block = f"""
            Title: {title.group(2).strip() if title else "Unknown"}
            Author: {author.group(2).strip() if author else "Unknown"}
            Supervisor: {supervisor.group(2).strip() if supervisor else "Unknown"}
            Year: {year.group(1).strip() if year else "Unknown"}
            Publisher: {publisher.group(2).strip() if publisher else "Unknown"}
            URL: {url.group(1).strip() if url else "Not Found"}

            Abstract/Summary
            {abstract}

            FULL TEXT START
            {full_text}
            """

            metadata_texts.append(metadata_block)  
        except Exception as e:  
            print(f"[ERROR] Failed to process {pdf_path.name}: {e}")  

    return metadata_texts
