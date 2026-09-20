from langchain_docling import DoclingLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from dotenv import load_dotenv


load_dotenv()

#where the files are
#FILE_PATH = "C:\\Users\\hayes\\Documents\\whragpdffolder\\RAGPDF\\samplereq.pdf"
FILE_PATH = "C:\\Users\\hayes\\Documents\\whragpdffolder\\RAGPDF\\RCV_Academy_Onboarding.docx"

ragdoc = (DoclingLoader(file_path=FILE_PATH))

document= ragdoc.load()

#print(document)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
doc_splits = text_splitter.split_documents(document)
#print("The content of first split is: ", doc_splits[0])

embeddings = OpenAIEmbeddings (
    model="text-embedding-3-large"
)

vector_store = InMemoryVectorStore(
   embeddings
)

vector_store.add_documents(doc_splits)

#results = vector_store.similarity_search(
#    query="what is the mandatory training for week 1"
#)

#for doc in results:
#    print(doc.page_content)

retriever = vector_store.as_retriever(
    search_type = "mmr",
    research_kwargs={"k":2}
)

retriever.invoke("What is the mandatory training for week 1")