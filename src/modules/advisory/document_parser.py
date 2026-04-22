import io
import re
import fitz  # PyMuPDF
from docx import Document
from src.shared.utils.logger import logger

def clean_text(raw_text: str) -> str:
    """
    Cleans raw extracted text for optimal chunking and embedding.
    - Normalizes whitespace and excessive newlines.
    - Joins broken paragraphs/sentences.
    - Removes common artifacts like floating page numbers.
    """
    if not raw_text:
        return ""
        
    text = raw_text
    
    # Remove floating page numbers (e.g., matching a line with just a number or "Page X")
    text = re.sub(r'(?m)^\s*(Page\s*\d+|\d+)\s*$', '', text, flags=re.IGNORECASE)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Join broken sentences (lines ending without punctuation followed by lowercase letter or continuation)
    # This specifically looks for a newline that shouldn't be there
    text = re.sub(r'([^\.!\?:;])\n([a-z])', r'\1 \2', text)
    
    # Replace single newlines with spaces to form solid paragraphs (if they are not double newlines)
    # A common pattern is a newline followed by a word character
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    
    # Collapse 3+ newlines into exactly two (representing a paragraph break)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Strip leading/trailing whitespaces
    return text.strip()

def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    """
    Routes extraction logic based on file extension.
    Supported: .pdf, .docx, .txt, .md
    """
    ext = filename.lower().split('.')[-1]
    raw_text = ""
    
    try:
        if ext == 'pdf':
            # PyMuPDF requires stream or bytes
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                text_pages = []
                for page in doc:
                    # Remove Header and Footer by clipping top and bottom 8%
                    rect = page.rect
                    margin_top = rect.height * 0.08
                    margin_bottom = rect.height * 0.08
                    
                    clip_rect = fitz.Rect(
                        rect.x0, 
                        rect.y0 + margin_top, 
                        rect.x1, 
                        rect.y1 - margin_bottom
                    )
                    
                    # Extract text only within the clipped area
                    text_pages.append(page.get_text("text", clip=clip_rect))
                raw_text = "\n\n".join(text_pages)
                
        elif ext in ['docx', 'doc']:
            # python-docx reads from a BytesIO stream
            doc = Document(io.BytesIO(file_bytes))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            raw_text = "\n".join(paragraphs)
            
        elif ext in ['txt', 'md', 'markdown']:
            raw_text = file_bytes.decode('utf-8', errors='ignore')
            
        else:
            raise ValueError(f"Unsupported file extension: .{ext}")
            
    except Exception as e:
        logger.error(f"Failed to parse file '{filename}': {e}")
        raise ValueError(f"Failed to parse document: {e}")
        
    return clean_text(raw_text)
