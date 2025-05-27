import os
import fitz  # PyMuPDF
import re # For text splitting in chunk_text_by_paragraph

# Default minimum chunk length for the chunking function
DEFAULT_MIN_CHUNK_LENGTH = 400

def read_file_content(filepath: str) -> str | None:
    """
    Reads text content from a PDF or plain text file.

    Args:
        filepath: The path to the file.

    Returns:
        The extracted text content as a string, or None if reading fails.
    """
    # Check if the file exists at the given path.
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return None

    # Determine the file extension to select the appropriate reading method.
    file_ext = os.path.splitext(filepath)[1].lower()

    # Handle PDF files.
    if file_ext == '.pdf':
        try:
            print(f"Attempting to extract text from PDF: {filepath}")
            doc = fitz.open(filepath)  # Open the PDF document.
            full_text = ""
            num_pages = doc.page_count
            print(f"PDF has {num_pages} pages.")
            # Iterate through each page and extract text.
            for i, page in enumerate(doc):
                page_text = page.get_text("text")  # Get raw text from the page.
                if page_text:
                    # Strip leading/trailing whitespace from page text and add a newline.
                    full_text += page_text.strip() + "\n"
            doc.close()
            # If no text could be extracted (e.g., image-based PDF with no OCR), return None.
            if not full_text.strip():
                 print("Warning: No text could be extracted from the PDF.")
                 return None
            print(f"Successfully extracted text from PDF.")
            return full_text
        except fitz.errors.PasswordError:  # Specific error for password-protected PDFs.
             print(f"Error: The PDF file '{filepath}' is password-protected.")
             return None
        except Exception as e:  # Catch other potential errors during PDF processing.
            # Provide more specific feedback for common fitz errors.
            if "cannot open" in str(e).lower() or "format error" in str(e).lower():
                print(f"Error: Cannot open '{filepath}'. It might not be a valid PDF or is corrupted. Details: {e}")
            else:
                print(f"Error extracting text from PDF '{filepath}': {e}")
            return None
    # Handle plain text files ('.txt', '.md', or no extension).
    elif file_ext == '.txt' or file_ext == '.md' or file_ext == '': 
        try:
            print(f"Reading plain text file: {filepath}")
            # Attempt to read with UTF-8 encoding first.
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError as ude:
            # If UTF-8 fails, attempt fallback to latin-1 encoding.
            print(f"UTF-8 decoding failed for '{filepath}' ({ude}), trying latin-1 encoding...")
            try:
                with open(filepath, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception as e2: # Catch errors during latin-1 reading.
                 print(f"Error reading plain text file '{filepath}' with UTF-8 and latin-1: {e2}")
                 return None
        except Exception as e: # Catch other file reading errors.
            print(f"Error reading plain text file '{filepath}': {e}")
            return None
    else:
        # If the file extension is not supported.
        print(f"Error: Unsupported file type: {file_ext}. Please provide a PDF or TXT file.")
        return None

def chunk_text_by_paragraph(text: str, min_length: int = DEFAULT_MIN_CHUNK_LENGTH) -> list[str]:
    """
    Chunks the text by paragraphs and filters short chunks.
    Fallback to sentence splitting if paragraph splitting yields no results.

    Args:
        text: The text content to chunk.
        min_length: The minimum length for a chunk to be considered valid.

    Returns:
        A list of text chunks.
    """
    if not text: # Return empty list if input text is empty.
        return []

    # Primary splitting strategy: by double newlines, common for paragraphs.
    paragraphs = text.split('\n\n')
    
    # Heuristic: If double newline splitting yields few paragraphs (<10) 
    # AND the text contains many single newlines (>20),
    # it might be structured with single newlines as paragraph separators.
    if len(paragraphs) < 10 and text.count('\n') > 20: 
        print("Few paragraphs found with '\\n\\n', trying single '\\n' splits.")
        paragraphs = text.split('\n') # Re-split by single newlines.
    elif len(paragraphs) < 10: # Log if still few paragraphs (e.g., short text).
         print("Few paragraphs found even with single '\\n' splits or short text overall.")

    # Filter out paragraphs that are shorter than min_length after stripping whitespace.
    chunks = [p.strip() for p in paragraphs if len(p.strip()) >= min_length]

    # Fallback to sentence splitting if no valid chunks were found by paragraph methods
    # AND the total text length itself is sufficient to potentially form a chunk.
    if not chunks and len(text.strip()) >= min_length:
         print(f"Warning: No text chunks found by paragraph splitting/filtering (min length {min_length}). Attempting sentence splitting.")
         # Basic sentence splitting using regex (splits after '.', '?', '!' followed by space).
         sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text.strip())
         current_chunk = ""
         for sentence in sentences:
             # Combine sentences until the current_chunk reaches min_length.
             if len(current_chunk) + len(sentence) + 1 >= min_length:
                 if current_chunk: # Add the formed chunk if it's not empty.
                    chunks.append(current_chunk.strip())
                 current_chunk = sentence + " " # Start a new chunk with the current sentence.
             else:
                 current_chunk += sentence + " " # Append sentence to the current chunk.
         
         # Add the last accumulated chunk if it's valid.
         if current_chunk and len(current_chunk.strip()) >= min_length: 
              chunks.append(current_chunk.strip())
         # If the last piece is too short but other chunks exist, append it to the last valid chunk.
         elif current_chunk and chunks: 
              chunks[-1] += " " + current_chunk.strip()

    # Final check: if no chunks were formed by any method.
    if not chunks:
        # If the original text itself (stripped) meets min_length, return it as a single chunk.
        # This handles cases where the text is a single block of text without many internal separators
        # but is long enough to be considered a chunk.
        if len(text.strip()) >= min_length:
            print(f"Warning: No valid text chunks found after splitting. Using the entire text as one chunk.")
            return [text.strip()]
        else:
            # If even the original text is too short, return an empty list.
            print(f"Warning: No valid text chunks found after splitting/filtering (min length {min_length}).")
            return []

    print(f"Chunked text into {len(chunks)} sections (min length {min_length}).")
    return chunks
