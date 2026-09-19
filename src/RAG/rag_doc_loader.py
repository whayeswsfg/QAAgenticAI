from langchain_docling import DoclingLoader
from dotenv import load_dotenv


load_dotenv()

#where the files are
FILE_PATH = "C:\\Users\\hayes\\Documents\\whragpdffolder\\RAGPDF\\RCV_Academy_HR_Manual.pdf"

ragdoc = (DoclingLoader(file_path=FILE_PATH))

document= ragdoc.load()

print(document)