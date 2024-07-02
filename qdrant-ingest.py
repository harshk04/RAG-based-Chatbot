from langchain_qdrant import Qdrant
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


def load_and_split_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    return texts

texts_u1 = load_and_split_pdf("Backend(LLM)-Assignment/goog-10-k-2023 (1).pdf")
texts_u2 = load_and_split_pdf("Backend(LLM)-Assignment/tsla-20231231-gen.pdf")
texts_u3 = load_and_split_pdf("Backend(LLM)-Assignment/uber-10-k-2023.pdf")


all_texts = texts_u1 + texts_u2 + texts_u3


model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


url = "http://localhost:6333"
qdrant = Qdrant.from_documents(
    all_texts,
    embeddings,
    url=url,
    prefer_grpc=False,
    collection_name="chatbot"
)

print("Collection successfully created!")
