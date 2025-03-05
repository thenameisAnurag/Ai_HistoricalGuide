# from langchain.document_loaders import DirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# data_path="E:\Ai Muesuem Guide\Books"
# def load_documents():
#     loader=DirectoryLoader(data_path,gloab="*.md")
#     documents=loader.load()
#     return documents 

# load_documents()
# Data Load -> Data Split -> Embedding -> Store
from langchain.document_loaders import PyPDFLoader
loader=PyPDFLoader('E:\Ai Muesuem Guide\Books\TheLifeofShivajiMaharaj_10252256.pdf')
documents=loader.load()
documents


from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter=RecursiveCharacterTextSplitter(chunk)