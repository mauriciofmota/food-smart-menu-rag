"""
================================================================================
ARQUITETURA DO GERENCIADOR DE BANCO VETORIAL (VECTOR STORE MANAGER)
================================================================================

Objetivo:
Este módulo é responsável por carregar os dados textuais estruturados (JSON),
transformá-los em representações vetoriais semânticas e disponibilizar um
sistema de recuperação eficiente (Retrieval) baseado no ChromaDB e Embeddings
do HuggingFace.

Fluxo de Execução:
1. Carregamento e Enriquecimento (`load_menu_data`):
   - Lê o arquivo JSON contendo os pratos e metadados.
   - Converte cada prato em um documento padronizado do LangChain (`Document`),
     construindo uma string textual rica para o modelo de embedding capturar
     contexto (restaurante, nome, descrição, preço, tempo de entrega e tags).
   - Preserva os atributos originais estruturados no dicionário de metadados.

2. Vetorização e Indexação (`initialize_vector_store`):
   - Utiliza o modelo leve e eficiente `all-MiniLM-L6-v2` da HuggingFace.
   - Instancia o ChromaDB em memória (`Chroma.from_documents`), criando o
     espaço vetorial para buscas por similaridade por cosseno.

3. Recuperação de Contexto (`get_retriever`):
   - Expõe uma interface limpa de `retriever` configurada com o parâmetro `k`
     (número de documentos mais relevantes retornados por consulta), pronta
     para ser integrada diretamente no pipeline LCEL do agente RAG.


OBS: os comentários e parâmetros estão em inglês para manter compatibilidade com a documentação oficial do LangChain e Ollama, além de
ser um padrão de mercado para facilitar futuras integrações com outros LLMs ou frameworks.
================================================================================
"""

import json
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

class VectorStoreManager:
    def __init__(self, json_path: str = "data/menu_mock.json"):
        # Initialize HuggingFace model for semantic representation
        self.json_path = json_path
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store = None

    def load_menu_data(self) -> list[Document]:
        """Loads JSON mock data and converts each dish into a LangChain Document."""
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"Menu data file not found at: {self.json_path}")

        with open(self.json_path, "r", encoding="utf-8") as file:
            dishes = json.load(file)

        documents = []
        for dish in dishes:
            # Create a rich textual representation for semantic embedding
            content = (
                f"Restaurant: {dish['restaurant']} | "
                f"Dish: {dish['name']} | "
                f"Description: {dish['description']} | "
                f"Price: ${dish['price']:.2f} | "
                f"Delivery Time: {dish['delivery_time_mins']} mins | "
                f"Tags: {', '.join(dish['tags'])}"
            )
            
            # Store original attributes in metadata for later retrieval
            metadata = {
                "id": dish["id"],
                "restaurant": dish["restaurant"],
                "name": dish["name"],
                "price": dish["price"],
                "delivery_time_mins": dish["delivery_time_mins"],
                "tags": ", ".join(dish["tags"])
            }
            
            documents.append(Document(page_content=content, metadata=metadata))

        return documents

    def initialize_vector_store(self) -> Chroma:
        """Initializes ChromaDB in memory and embeds the menu documents."""
        documents = self.load_menu_data()
        
        # In-memory vector store
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name="restaurant_menu"
        )
        return self.vector_store

    def get_retriever(self, k: int = 2):
        """Returns a retriever object configured to return top-k matches."""
        if not self.vector_store:
            self.initialize_vector_store()
        return self.vector_store.as_retriever(
            search_kwargs={"k": k}
        )