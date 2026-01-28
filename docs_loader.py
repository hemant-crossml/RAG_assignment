from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader


def load_documents(data_dir: str):
    loader = DirectoryLoader(
        data_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    return loader.load()
