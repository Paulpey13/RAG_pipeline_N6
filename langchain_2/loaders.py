from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredExcelLoader,
)
from pathlib import Path

def load_file(file_path: Path):
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(str(file_path))
        return loader.load()
    elif ext == ".txt":
        loader = TextLoader(str(file_path))
        return loader.load()
    elif ext == ".docx":
        loader = Docx2txtLoader(str(file_path))
        return loader.load()
    elif ext == ".csv":
        loader = CSVLoader(str(file_path))
        return loader.load()
    elif ext in [".xls", ".xlsx"]:
        loader = UnstructuredExcelLoader(str(file_path))
        return loader.load()
    else:
        return []
