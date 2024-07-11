#Complete Code for db_init.py
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.document_loaders import PyMuPDFLoader

# documents=PyMuPDFLoader(file_path=file_path).load()

 
# text_splitter = RecursiveCharacterTextSplitter(
#     separators=["\n"],
#     chunk_size=512,
#     chunk_overlap=100
# )

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
)

 
# loader = TextLoader("cleaned_extracted_text.txt",encoding = 'UTF-8')
loader = PyMuPDFLoader("swx1.pdf")
docs = loader.load_and_split(
    text_splitter=text_splitter
)
 
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
 
db = Chroma.from_documents(
    docs,
    embedding=embeddings,
    persist_directory="data1"
)

