
import fitz
import pytesseract
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)
from PIL import Image
from io import BytesIO

file_path = "./Documents/L3 Natural resources.pdf"

# Windows par zarurat pade toh Tesseract ka path set karo:
# pytesseract.pytesseract.tesseract_cmd = (
#     r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# )

pdf = fitz.open(file_path)

print("Total PDF pages:", len(pdf))

for page_index in range(7, 9):
    page = pdf[page_index]

    # PDF page ko image mein render karo
    pix = page.get_pixmap(
        matrix=fitz.Matrix(2, 2),
        alpha=False
    )

    image = Image.open(BytesIO(pix.tobytes("png")))

    # Image se text extract karo
    extracted_text = pytesseract.image_to_string(
        image,
        lang="eng"
    )

    print(f"\n--- Page {page_index + 1} ---")
    print(extracted_text[:1500])
