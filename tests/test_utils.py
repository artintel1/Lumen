import unittest
import os
import sys

# Add the project root to the Python path to allow importing 'utils'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from utils import read_file_content, chunk_text_by_paragraph, DEFAULT_MIN_CHUNK_LENGTH

class TestReadFileContent(unittest.TestCase):

    SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "sample_files")
    TXT_SAMPLE_PATH = os.path.join(SAMPLES_DIR, "sample.txt")
    PDF_SAMPLE_PATH = os.path.join(SAMPLES_DIR, "sample.pdf")
    EMPTY_TXT_PATH = os.path.join(SAMPLES_DIR, "empty.txt")
    LATIN1_TXT_PATH = os.path.join(SAMPLES_DIR, "latin1_sample.txt")
    NON_EXISTENT_PATH = os.path.join(SAMPLES_DIR, "non_existent_file.xyz")

    def test_read_sample_txt(self):
        """Test reading a standard UTF-8 text file."""
        expected_content = "This is a sample text file.\nIt has multiple lines.\nHello, world!"
        # The read_file_content for text files might add a trailing newline if the file has one,
        # or if the reading process adds one. Standard open().read() doesn't add one if not present.
        # The sample.txt was created without a trailing newline on the last line of content.
        # Let's ensure the expected content matches exactly how it was created.
        # The create_file_with_block adds a newline at the end of the *entire* block.
        # So, the actual file content will be "Hello, world!\n"
        expected_content_actual = "This is a sample text file.\nIt has multiple lines.\nHello, world!\n"
        content = read_file_content(self.TXT_SAMPLE_PATH)
        self.assertIsNotNone(content)
        self.assertEqual(content, expected_content_actual)

    def test_read_sample_pdf(self):
        """Test reading a simple PDF file."""
        # Text extraction from PDF can be tricky. We'll check for presence of key phrases.
        # The PDF was created with "This is a sample PDF file. Hello, PDF world!"
        # reportlab's drawString doesn't typically process \n within the string for multiple lines.
        # The read_file_content function adds "\n" after each page's text.
        # For a single page PDF, it should be "text_from_page_1\n"
        expected_substrings = ["This is a sample PDF file.", "Hello, PDF world!"]
        content = read_file_content(self.PDF_SAMPLE_PATH)
        self.assertIsNotNone(content, "Reading PDF should not return None")
        # Extracted text from PDF often has extra spaces or formatting quirks.
        # Making the check robust to this.
        normalized_content = " ".join(content.split()) if content else ""
        
        # For reportlab, text is often directly concatenated.
        # The actual text from PDF should be "This is a sample PDF. Hello PDF world.\n"
        expected_pdf_content = "This is a sample PDF. Hello PDF world.\n"
        self.assertEqual(content, expected_pdf_content)
        # Also check for key phrases to be robust against minor spacing/newline variations if needed,
        # though for this simple PDF, exact match should work.
        self.assertIn("This is a sample PDF", content)
        self.assertIn("Hello PDF world", content)

        # Verify directly with fitz to ensure consistency if test fails
        import fitz
        doc = fitz.open(self.PDF_SAMPLE_PATH)
        pdf_text_content_from_test = ""
        for page in doc:
            page_text = page.get_text("text")
            if page_text: # Ensure page_text is not None
                 pdf_text_content_from_test += page_text.strip() + "\n"
        doc.close()
        self.assertEqual(content, pdf_text_content_from_test)


    def test_read_non_existent_file(self):
        """Test reading a non-existent file."""
        content = read_file_content(self.NON_EXISTENT_PATH)
        self.assertIsNone(content, "Reading non-existent file should return None")

    def test_read_empty_txt(self):
        """Test reading an empty text file."""
        # empty.txt was created with no content. create_file_with_block might add a newline.
        # If create_file_with_block adds a newline to an empty block, content is "\n".
        # If it creates a truly empty 0-byte file, content is "".
        # The read_file_content function uses open().read().
        # Let's check actual file size of empty.txt, if it's 0 bytes, expected is ""
        # If it's 1 byte (just a newline), expected is "\n"
        
        # Re-evaluating: create_file_with_block for an empty content string creates an empty file (0 bytes).
        expected_content = ""
        content = read_file_content(self.EMPTY_TXT_PATH)
        self.assertIsNotNone(content) # Should return empty string, not None
        self.assertEqual(content, expected_content)

    def test_read_latin1_sample_txt(self):
        """Test reading a latin-1 encoded text file."""
        # The file was created with "Este es un ejemplo con caracteres latinos: ñ, é, ç, ü.\n¡Hola, mundo!\n"
        # (create_file_with_block adds a newline)
        expected_content = "Este es un ejemplo con caracteres latinos: ñ, é, ç, ü.\n¡Hola, mundo!\n"
        content = read_file_content(self.LATIN1_TXT_PATH)
        self.assertIsNotNone(content)
        self.assertEqual(content, expected_content)
        # Ensure it was actually decoded correctly (Python strings are UTF-8 internally)
        self.assertIn("ñ", content)
        self.assertIn("é", content)
        self.assertIn("ç", content)
        self.assertIn("ü", content)

    def test_read_unsupported_file_type(self):
        """Test reading an unsupported file type (e.g., a binary file or different extension)."""
        # Create a dummy .zip file for testing this
        dummy_zip_path = os.path.join(self.SAMPLES_DIR, "dummy.zip")
        with open(dummy_zip_path, "wb") as f:
            f.write(b"PK\x03\x04...") # Minimal zip header
        
        content = read_file_content(dummy_zip_path)
        self.assertIsNone(content, "Reading an unsupported file type should return None")
        
        os.remove(dummy_zip_path) # Clean up

    # Note: Testing for password-protected PDFs is omitted due to the complexity
    # of reliably creating such a file programmatically in this environment.
    # The read_file_content function includes a try-except for fitz.errors.PasswordError.

class TestChunkTextByParagraph(unittest.TestCase):
    ML = DEFAULT_MIN_CHUNK_LENGTH # Alias for convenience

    def _generate_long_string(self, length, char_sequence="word "):
        """Helper to generate a string of a specific approximate length."""
        return (char_sequence * (length // len(char_sequence) + 1))[:length]

    def test_empty_input(self):
        """Test with an empty string input."""
        self.assertEqual(chunk_text_by_paragraph(""), [])

    def test_basic_paragraph_splitting(self):
        """Test basic paragraph splitting with \\n\\n, all paragraphs long enough."""
        para1 = self._generate_long_string(self.ML + 50, "para1 ")
        para2 = self._generate_long_string(self.ML + 60, "para2 ")
        text = f"{para1}\n\n{para2}"
        chunks = chunk_text_by_paragraph(text)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0], para1.strip()) # Strip expectation
        self.assertEqual(chunks[1], para2.strip()) # Strip expectation

    def test_short_paragraphs_filtered(self):
        """Test that short paragraphs are filtered out."""
        long_para = self._generate_long_string(self.ML + 50, "long ")
        short_para = self._generate_long_string(self.ML - 50, "short ") # This will be filtered
        text = f"{long_para}\n\n{short_para}\n\n{long_para}" # Only long_para should remain
        chunks = chunk_text_by_paragraph(text)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0], long_para.strip()) # Strip expectation
        self.assertEqual(chunks[1], long_para.strip()) # Strip expectation

    def test_single_newline_splitting(self):
        """Test splitting by single \\n when \\n\\n is not prevalent."""
        # Create text that has many single newlines but few double newlines
        # The heuristic is: len(paragraphs) < 10 and text.count('\n') > 20
        para1 = self._generate_long_string(self.ML + 10, "s_line1 ")
        para2 = self._generate_long_string(self.ML + 20, "s_line2 ")
        para3 = self._generate_long_string(self.ML + 30, "s_line3 ")
        # Join with single newlines, ensuring more than 20 single newlines
        # and fewer than 10 double newlines (0 in this case)
        text = f"{para1}\n" * 8 + f"{para2}\n" * 8 + f"{para3}\n" * 8
        # This creates 3 "logical" paragraphs if we consider the source, but they are very long.
        # The splitting logic will split by '\n' if heuristic is met.
        # Let's test with text structured as blocks separated by single newlines.
        text_single_newline = f"{para1}\n{para2}\n{para3}" # 3 paragraphs, 2 single \n
        # To trigger the heuristic: text.count('\n') > 20
        # This test case needs to be carefully crafted.
        # The heuristic is: if default \n\n split yields < 10 paras, AND text.count('\n') > 20
        # then it re-splits by \n.
        
        # Scenario 1: Heuristic IS met.
        single_nl_para = self._generate_long_string(self.ML + 5, "snl_para ")
        # Create more than 20 single newlines, with less than 10 paragraphs if split by \n\n
        text_for_heuristic = (single_nl_para.strip() + "\n") * 25 # Use strip to avoid double spaces from helper
        
        chunks = chunk_text_by_paragraph(text_for_heuristic.strip()) # Strip final newline
        self.assertEqual(len(chunks), 25) # Expects each line to be a chunk
        self.assertTrue(all(c == single_nl_para.strip() for c in chunks))

        # Scenario 2: Heuristic NOT met by \n\n (e.g. already >10 paras by \n\n)
        # This would just use the default \n\n split.
        double_nl_para = self._generate_long_string(self.ML + 5, "dnl_para ")
        text_many_double_nl = (double_nl_para.strip() + "\n\n") * 15
        chunks_double = chunk_text_by_paragraph(text_many_double_nl.strip()) # Strip final newlines
        self.assertEqual(len(chunks_double), 15)
        self.assertTrue(all(c == double_nl_para.strip() for c in chunks_double))


    def test_sentence_splitting_fallback_long_text_no_paragraphs(self):
        """Test sentence splitting for long text with no paragraph breaks."""
        sentence = "This is a sentence with about ten words. "
        # Approx 8 sentences to make a chunk of length ML (400).
        # Create text for about 2.5 chunks.
        num_sentences_for_one_chunk = (self.ML // len(sentence)) + 1
        total_sentences = int(num_sentences_for_one_chunk * 2.5)
        long_text_no_paragraphs = (sentence * total_sentences).strip()
        
        chunks = chunk_text_by_paragraph(long_text_no_paragraphs)
        
        # Expected number of chunks:
        # Total length / ML, or how sentence combination works.
        # E.g., if ML=400, sentence=50. 8 sentences = 400.
        # 2.5 * 8 = 20 sentences. Should give 2 full chunks and one partial if combined.
        # The current logic combines greedily.
        # 20 sentences * 50char/sentence = 1000 chars.
        # Should be 2 chunks if sentence combination is precise. (1000 / 400 = 2.5)
        # Chunk 1: sentences 1-8 (400 chars)
        # Chunk 2: sentences 9-16 (400 chars)
        # Remainder: sentences 17-20 (200 chars) - this will be appended to chunk 2 if logic is `chunks[-1] += ...`
        # Let's re-check utils.py: `elif current_chunk and chunks: chunks[-1] += " " + current_chunk.strip()`
        # This means the last short bit is appended. So, 2 chunks, second one longer.
        
        self.assertTrue(len(chunks) > 0, "Should produce chunks for long text.")
        if len(chunks) > 0:
            self.assertTrue(len(chunks[0]) >= self.ML, f"First chunk length {len(chunks[0])} vs min {self.ML}")
        # For the last chunk, it might be shorter if it's the very last part of text
        # or longer if short remainders were appended.
        # All chunks except possibly the last one (if it didn't have enough to form fully) should be >= ML
        for i in range(len(chunks) -1 ): # Check all but last chunk
             self.assertTrue(len(chunks[i]) >= self.ML, f"Chunk {i} length {len(chunks[i])} vs min {self.ML}")
        
        # Sum of chunk lengths should be close to original text length (minus spaces for joining)
        total_chunked_length = sum(len(c) for c in chunks)
        self.assertTrue(abs(total_chunked_length - len(long_text_no_paragraphs)) < len(chunks) * 2, "Total length of chunks is too different from original")


    def test_short_input_less_than_min_length(self):
        """Test input text shorter than DEFAULT_MIN_CHUNK_LENGTH."""
        short_text = self._generate_long_string(self.ML - 50, "short_text ")
        chunks = chunk_text_by_paragraph(short_text)
        # Expected behavior: if len(text.strip()) < min_length, returns []
        self.assertEqual(chunks, [])

    def test_input_just_over_min_length_as_single_chunk(self):
        """Test input text just over min_length, becoming a single chunk."""
        text = self._generate_long_string(self.ML + 50, "single_chunk_text ")
        # This text has no paragraph breaks, so it should fallback to sentence splitting,
        # and then if no sentences make a chunk that meets min_length, 
        # the whole thing becomes one chunk if len(text.strip()) >= min_length.
        chunks = chunk_text_by_paragraph(text)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text.strip()) # chunk_text_by_paragraph strips the final chunk

    def test_sentence_fallback_from_short_paragraphs(self):
        """Test sentence fallback is triggered when initial paragraphs are all too short."""
        # Text is long enough, but all paragraphs are short.
        # Total length = (50 chars + 2 for \n\n) * 20 paragraphs = 1040 chars.
        # This is > ML (400).
        # Paragraph splitting will yield 0 valid chunks.
        # Sentence splitting should then occur on the entire text.
        # Sentences here are effectively the "short paragraphs".
        short_para_content = self._generate_long_string(50, "tiny_para_sentence. ")
        text_many_short_paras = (short_para_content + "\n\n") * 20 
        text_many_short_paras = text_many_short_paras.strip() # Remove final \n\n for accurate length count

        chunks = chunk_text_by_paragraph(text_many_short_paras)
        
        # Expected: sentence splitter combines these "tiny_para_sentence." blocks.
        # Each block is 50 chars. About 8 such blocks to make one 400-char chunk.
        # 20 blocks total. So, 20 * 50 = 1000 chars. Expected: 2 chunks.
        # (1000 / 400 = 2.5, last part appended)
        self.assertTrue(len(chunks) > 0, "Sentence fallback should produce chunks.")
        # For this specific input, it's likely to be 2 chunks if ML=400
        # (e.g. 8*50=400, 8*50=400, remaining 4*50=200 appended to 2nd chunk)
        # or 1 chunk if the sentence splitting is very aggressive in combining.
        # Given the logic, it should be 2 chunks. The last one will be longer.
        # chunk1 = 8 * "tiny_para_sentence. " (joined)
        # chunk2 = 12 * "tiny_para_sentence. " (joined)
        expected_num_chunks = (len(short_para_content) * 20) // self.ML
        if (len(short_para_content) * 20) % self.ML > 0 and expected_num_chunks > 0 : # if there's a remainder for the last chunk
             pass # it will be appended, so num_chunks doesn't increase
        elif expected_num_chunks == 0 and (len(short_para_content) * 20) >= self.ML:
             expected_num_chunks = 1

        self.assertEqual(len(chunks), expected_num_chunks if expected_num_chunks > 0 else 1 if (len(short_para_content) * 20) >= self.ML else 0 )


    def test_custom_min_length(self):
        """Test with a custom min_length argument."""
        custom_ml = 50
        para1 = self._generate_long_string(custom_ml + 10, "custom1 ")
        para2 = self._generate_long_string(custom_ml - 10, "custom_short ") # Should be filtered
        para3 = self._generate_long_string(custom_ml + 20, "custom3 ")
        text = f"{para1}\n\n{para2}\n\n{para3}"
        chunks = chunk_text_by_paragraph(text, min_length=custom_ml)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0], para1.strip()) # Strip expectation
        self.assertEqual(chunks[1], para3.strip()) # Strip expectation

if __name__ == '__main__':
    unittest.main()
