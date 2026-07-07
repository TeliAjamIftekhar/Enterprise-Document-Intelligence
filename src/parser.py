"""
PDF Parser

Responsible for:
1. Opening PDF
2. Extracting text
3. Extracting images
4. Returning structured page data
"""

import fitz


class PDFParser:

    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)

    def get_total_pages(self):
        return len(self.doc)

    def parse_pages(self, start_page=0, end_page=None):

        if end_page is None:
            end_page = len(self.doc)

        pages = []

        for page_no in range(start_page, end_page):

            page = self.doc.load_page(page_no)

            text = page.get_text("text")

            images = page.get_images(full=True)

            pages.append({

                "page": page_no + 1,

                "text": text,

                "images": images

            })

        return pages