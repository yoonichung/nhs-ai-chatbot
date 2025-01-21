from langchain_community.document_loaders import PyPDFDirectoryLoader


def load_pdf(pdf_path):
    pdf_loader = PyPDFDirectoryLoader(pdf_path)

    return pdf_loader.load()


if __name__ == "__main__":
    pdf_path = "/Users/yooni/Repos/nhs-ai-chatbot/sample_pdfs/VV open Surgery.pdf"
    doc = load_pdf(pdf_path)
    print(doc)

# TODO currently only shows empty list - add try-except block to catch errors
