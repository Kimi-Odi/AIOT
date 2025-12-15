# pdf_export.py

from reportlab.pdfgen import canvas
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
import textwrap
import os
from matplotlib import font_manager
from pdf_styles import *

def _register_font():
    """
    Register a CJK-capable font. Try system TTFs via matplotlib, then CID fonts,
    and finally fall back to Helvetica (ASCII only).
    """
    # Try common system fonts (Windows / Noto / SimSun / MingLiU / JhengHei)
    candidate_names = [
        "Microsoft JhengHei",
        "PMingLiU",
        "MingLiU",
        "SimSun",
        "Noto Sans CJK TC",
        "Noto Sans CJK SC",
    ]
    font_path = None
    for name in candidate_names:
        try:
            path = font_manager.findfont(name, fallback_to_default=False)
            if path and os.path.isfile(path):
                font_path = path
                break
        except Exception:
            continue
    
    if font_path:
        try:
            pdfmetrics.registerFont(TTFont(FONT_NAME, font_path))
            # Try to find a bold version
            try:
                bold_path = font_manager.findfont(name + " Bold", fallback_to_default=False)
                if bold_path and os.path.isfile(bold_path):
                     pdfmetrics.registerFont(TTFont(FONT_BOLD_NAME, bold_path))
                else:
                    # Fallback to regular if bold not found
                    pdfmetrics.registerFont(TTFont(FONT_BOLD_NAME, font_path))
            except:
                pdfmetrics.registerFont(TTFont(FONT_BOLD_NAME, font_path))

            return FONT_NAME, FONT_BOLD_NAME
        except Exception:
            pass


    # Try built-in CID fonts (should be supported by most PDF viewers)
    for cid_name in ["STSong-Light", "MSung-Light"]:
        try:
            pdfmetrics.registerFont(UnicodeCIDFont(cid_name))
            return cid_name, cid_name # Use same for bold
        except Exception:
            continue

    # Last resort: ASCII only
    return "Helvetica", "Helvetica-Bold"


FONT_NAME, FONT_BOLD_NAME = _register_font()


class PDFReport:
    def __init__(self, path, page_title="AI 面試報告"):
        self.c = canvas.Canvas(path, pagesize=A4)
        self.width, self.height = A4
        self.margin_x = 20 * mm
        self.margin_y = 20 * mm
        self.y = self.height - self.margin_y
        self.page_number = 1
        self.page_title = page_title
        self._draw_footer()

    def _draw_header(self):
        self.c.setFont(FONT_BOLD_NAME, H1_SIZE)
        self.c.setFillColor(PRIMARY_COLOR)
        self.c.drawString(self.margin_x, self.height - self.margin_y + 10, self.page_title)
        self.y -= 30

    def _draw_footer(self):
        self.c.setFont(FONT_NAME, 8)
        self.c.setFillColor(SECONDARY_COLOR)
        self.c.drawCentredString(self.width / 2, self.margin_y / 2, f"第 {self.page_number} 頁")

    def _new_page(self):
        self.c.showPage()
        self.page_number += 1
        self.y = self.height - self.margin_y
        self._draw_footer()
        # self._draw_header() # Optional: add header to new pages

    def _check_page_break(self, height_needed):
        if self.y - height_needed < self.margin_y:
            self._new_page()

    def draw_h1(self, text):
        self._check_page_break(30)
        self.c.setFont(FONT_BOLD_NAME, H1_SIZE)
        self.c.setFillColor(PRIMARY_COLOR)
        self.c.drawString(self.margin_x, self.y, text)
        self.y -= H1_SIZE * LINE_HEIGHT

    def draw_h2(self, text):
        self._check_page_break(25)
        self.c.setFont(FONT_BOLD_NAME, H2_SIZE)
        self.c.setFillColor(PRIMARY_COLOR)
        self.c.drawString(self.margin_x, self.y, text)
        self.y -= H2_SIZE * LINE_HEIGHT

    def draw_h3(self, text):
        self._check_page_break(20)
        self.c.setFont(FONT_BOLD_NAME, H3_SIZE)
        self.c.setFillColor(SECONDARY_COLOR)
        self.c.drawString(self.margin_x, self.y, text)
        self.y -= H3_SIZE * LINE_HEIGHT
    
    def draw_paragraph(self, text):
        self.c.setFont(FONT_NAME, BODY_SIZE)
        self.c.setFillColorRGB(0, 0, 0)
        lines = textwrap.wrap(text, width=80) or [""]
        for line in lines:
            self._check_page_break(BODY_SIZE * LINE_HEIGHT)
            self.c.drawString(self.margin_x, self.y, line)
            self.y -= BODY_SIZE * LINE_HEIGHT
        self.y -= 10 # Extra space after paragraph

    def draw_list_item(self, text):
        self.c.setFont(FONT_NAME, BODY_SIZE)
        self.c.setFillColorRGB(0, 0, 0)
        lines = textwrap.wrap(text.lstrip("- "), width=75) or [""]
        self._check_page_break(len(lines) * BODY_SIZE * LINE_HEIGHT)
        self.c.drawString(self.margin_x, self.y, "•")
        for i, line in enumerate(lines):
            self.c.drawString(self.margin_x + 5 * mm, self.y, line)
            if i < len(lines) - 1:
                self.y -= BODY_SIZE * LINE_HEIGHT

        self.y -= BODY_SIZE * LINE_HEIGHT

    def draw_image(self, img_path):
        if not img_path or not os.path.exists(img_path):
            return
        
        img_width = self.width - 2 * self.margin_x
        img_height = img_width * 0.75
        self._check_page_break(img_height + 20)
        
        self.y -= 10 # Space before image
        img = ImageReader(img_path)
        self.c.drawImage(img, self.margin_x, self.y - img_height, width=img_width, height=img_height, preserveAspectRatio=True, mask='auto')
        self.y -= (img_height + 20)

    def save(self):
        self.c.save()


def export_pdf(path, markdown_text, image_paths=None):
    """
    將 Markdown 轉成更美觀的 PDF
    """
    report = PDFReport(path)
    
    lines = markdown_text.split("\n") if markdown_text else []
    if not lines:
        lines = ["（本次沒有可匯出的內容）"]

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("# "):
            report.draw_h1(line[2:])
        elif line.startswith("## "):
            report.draw_h2(line[3:])
        elif line.startswith("### "):
            report.draw_h3(line[4:])
        elif line.startswith("- "):
            report.draw_list_item(line)
        else:
            report.draw_paragraph(line)

    if image_paths:
        for img_path in image_paths:
            report.draw_image(img_path)

    report.save()
