import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

## load the files

def load_documents(docs_path= "docs"):
  """Load the documents"""
  if not os.path.exists(docs_path):
    raise FileNotFoundError(f"Files does not exist at location {docs_path}")

  #initialize the loader
  loader = DirectoryLoader(
    path= docs_path,
    glob="*.txt",
    loader_cls=TextLoader
  )
  documents= loader.load()


  if len(documents)==0:
    raise FileNotFoundError(f"Please add the documents at {docs_path}")
  for i, doc in enumerate(documents[:2]):  # Show first 2 documents
        print(f"\nDocument {i+1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")
        print(f"  metadata: {doc.metadata}")
  return documents


if __name__ =="__main__":
  # main()
  docs_path = "docs"
  documents = load_documents(docs_path)
