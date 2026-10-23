
import fitz

file_path = "./Documents/L3 Natural resources.pdf"
pdf = fitz.open(file_path)

print("Total pages:", len(pdf))

for index, page in enumerate(pdf):
    text = page.get_text("text").strip()
    images = page.get_images(full=True)
    words = page.get_text("words")

    print(f"\nPage {index + 1}")
    print("  Extracted characters:", len(text))
    print("  Extracted words:", len(words))
    print("  Embedded images:", len(images))

    if text:
        print("  Text preview:", repr(text[:150]))
    else:
        print("  Text preview: No extractable text")

    if not text and images:
        print("  Status: Image-only candidate")
    elif text and images:
        print("  Status: Mixed content candidate")
    elif text:
        print("  Status: Text extractable")
    else:
        print("  Status: Needs visual inspection")
