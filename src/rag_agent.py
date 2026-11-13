"""
================================================================================
ARQUITETURA DO AGENTE RAG CONVERSACIONAL
================================================================================

Objetivo:
Este módulo implementa o núcleo de inteligência artificial do sistema, unindo
Busca Semântica (Retrieval-Augmented Generation) com Memória Conversacional
de múltiplos turnos, rodando 100% localmente via Ollama (Qwen 2.5 - 3B).

Fluxo de Execução (Pipeline LCEL):
1. Inicialização (`__init__`):
   - Conecta ao LLM local (`ChatOllama`) com temperatura baixa (`0.3`) para
     priorizar precisão factual e evitar alucinações.
   - Instancia o `VectorStoreManager` (ChromaDB + HuggingFace Embeddings) e
     configura o `retriever` para buscar os `k` pratos mais relevantes.
   - Inicializa um dicionário em memória (`self.store`) para isolar o histórico
     de conversa por `session_id`.

2. Estágio de Recuperação (`retrieval_stage`):
   - Recebe o dicionário de entrada contendo a pergunta atual (`question`) e o
     histórico acumulado (`chat_history`).
   - Extrai apenas a string da pergunta e consulta o banco vetorial, formatando
     os documentos retornados em um bloco único de texto (`context`).

3. Engenharia de Prompt e Guardrails (`prompt`):
   - Injeta as regras estritas de negócio (System Prompt): proibição de inventar
     pratos, bloqueio imediato de restaurantes fechados `(CLOSED)`, alertas
     obrigatórios de alérgenos e verificação de promoções expiradas.
   - Insere o histórico (`MessagesPlaceholder`) para permitir refinamentos
     naturais entre turnos (ex: "tem algo mais barato que essa opção?").

4. Geração e Persistência de Estado (`RunnableWithMessageHistory`):
   - Envia o prompt montado ao Qwen 2.5, converte a saída em texto limpo
     (`StrOutputParser`) e salva automaticamente a pergunta do usuário e a
     resposta da IA no histórico da sessão correspondente.

OBS: os comentários e parâmetros estão em inglês para manter compatibilidade com a documentação oficial do LangChain e Ollama, além de
ser um padrão de mercado para facilitar futuras integrações com outros LLMs ou frameworks.
================================================================================
"""

from operator import itemgetter
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from src.vector_store import VectorStoreManager


class RAGAgent:
    def __init__(self, model_name: str = "qwen2.5:3b", k_results: int = 3):
        # Initialize local Ollama model.
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.3
        )
        
        # Initialize vector store manager and configure semantic retriever
        self.vector_manager = VectorStoreManager()
        self.retriever = self.vector_manager.get_retriever(k=k_results)
        
        # In-memory session store for multi-turn chat history
        self.store = {}
        
        self.chain = self._build_rag_chain()

    def _format_docs(self, docs) -> str:
        """Helper to format retrieved documents into a single context string."""
        return "\n\n".join(doc.page_content for doc in docs)

    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Retrieves or creates chat history for a specific session ID."""
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def _build_rag_chain(self):
        """Builds the LCEL RAG pipeline with conversation history and strict guardrails."""
        
        system_template = """You are an expert AI assistant for an intelligent food delivery platform. 
Your goal is to help users find the ideal dish based on their natural language request, dietary needs, or budget.

Strict Rules:
1. Use ONLY the provided menu context below to recommend dishes. Do not invent items.
2. Pay extreme attention to edge cases in the context:
   - If a restaurant is marked as (CLOSED), warn the user immediately and do not recommend it.
   - If an item mentions allergen warnings (e.g., peanuts, nuts, shellfish, lactose), explicitly caution the user.
   - If a promotion or code is expired, inform the user clearly.
3. If the context does not contain a matching dish, politely inform the user and suggest the closest available option.
4. Be concise, friendly, and structure your recommendations clearly (mention restaurant name, price, and why it fits).
5. Take previous conversation turns into account when refining choices (e.g., if the user asks for alternatives or cheaper options)."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "Menu Context:\n{context}\n\nUser Request:\n{question}")
        ])

        # Retrieval stage: routes the question to the vector store while preserving chat history
        retrieval_stage = {
            "context": lambda x: self._format_docs(self.retriever.invoke(x["question"])),
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"]
        }

        # LCEL pipeline
        rag_chain = (
            retrieval_stage
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # Core chain with session-based conversational memory - to update
        conversational_chain = RunnableWithMessageHistory(
            rag_chain,
            self._get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history",
        )
        
        return conversational_chain

    def ask(self, query: str, session_id: str = "default_user") -> str:
        """Invokes the conversational RAG chain with a session identifier."""
        return self.chain.invoke(
            {"question": query},
            config={"configurable": {"session_id": session_id}}
        )