from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

def create_simple_pdf(filepath, text_content):
    try:
        c = canvas.Canvas(filepath, pagesize=letter)
        c.drawString(100, 750, text_content)
        c.save()
        print(f"Successfully created PDF: {filepath}")
    except Exception as e:
        print(f"Error creating PDF {filepath}: {e}")
        # Ensure the file is not left in a corrupted state if possible
        import os
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == "__main__":
    # Ensure the target directory exists
    import os
    output_dir = "tests/sample_files"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    pdf_path = os.path.join(output_dir, "sample.pdf")
    content = "This is a sample PDF. Hello PDF world." # Simpler, single line content
    create_simple_pdf(pdf_path, content)
