from langchain_docling import DoclingLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


load_dotenv()

#where the files are
FILE_PATH = "C:\\Users\\hayes\\Documents\\whragpdffolder\\RAGPDF\\samplereq.pdf"
#FILE_PATH = "C:\\Users\\hayes\\Documents\\whragpdffolder\\RAGPDF\\RCV_Academy_Onboarding.docx"

ragdoc = (DoclingLoader(file_path=FILE_PATH))

document= ragdoc.load()

#print(document)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)
doc_splits = text_splitter.split_documents(document)
#print("The content of first split is: ", doc_splits[0])

embeddings = OpenAIEmbeddings (
    model="text-embedding-3-large"
)

vector_store = InMemoryVectorStore(  #we can also use MongoDB, Chroma, etc.
   embeddings
)

vector_store.add_documents(doc_splits)


myretriever = vector_store.as_retriever(
    search_type = "mmr", #mmr means maximal marginal relevance; balance simularity# of docs to return; with diversity among the selected results
    research_kwargs={"k":1}  #restrict output to 2 documents; fetch_k = amount of doucments to pass to MMR algorithm
)

#results=myretriever.invoke("What is the mandatory training for week 2")
#for doc in results:
#    print(doc.page_content)

llm = ChatOpenAI(model="gpt-5-nano")

#chat prompt template to formulate our chatprompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer user queries using only the provided context below: \n{context}"),
    ("human", "{question}")
])

chain = (
        {"context": myretriever,"question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

response = chain.invoke("What is hte mandatory training for week 4")

print(response)