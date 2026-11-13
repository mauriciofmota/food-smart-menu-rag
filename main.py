"""
================================================================================
FLUXO PRINCIPAL DE EXECUÇÃO (MAIN INTERFACE - FOOD SMART MENU ASSISTANT)
================================================================================

Objetivo:
Este script serve como a interface de entrada via terminal (CLI) para interagir
com o agente RAG conversacional. Ele gerencia o ciclo de vida da aplicação,
a inicialização do modelo local, a manutenção do estado da sessão e o loop
interativo de perguntas e respostas.

Fluxo de Execução:
1. Inicialização:
   - Exibe o banner de inicialização do sistema e instancia a classe `RAGAgent`
     carregando o Qwen 2.5 (3B) e o banco vetorial ChromaDB.
   - Define um identificador padrão de sessão (`session_id`) para isolar o histórico.

2. Loop Interativo (`while True`):
   - Captura a entrada em linguagem natural do usuário pelo terminal.
   - Trata comandos de saída (`exit`, `quit`).
   - Invoca o agente passando a query atual e o ID da sessão, exibindo a
     resposta estruturada com base no contexto do cardápio e nos guardrails.

3. Tratamento de Exceções:
   - Captura falhas críticas de execução (como a ausência do serviço Ollama
     rodando em segundo plano) e orienta o desenvolvedor.

     
OBS: os comentários e parâmetros estão em inglês para manter compatibilidade com a documentação oficial do LangChain e Ollama, além de
ser um padrão de mercado para facilitar futuras integrações com outros LLMs ou frameworks.
================================================================================
"""


from src.rag_agent import RAGAgent
# The robot emoji is used to indicate the AI assistant in the terminal output.
# Deal with the langchain deprecation warnings for a cleaner CLI experience. -todo
def main():
    print("=" * 60)
    print("Initializing Food Smart Menu & Intent Assistant (Local RAG v2)...")
    print("🤖 Loading Qwen 2.5 (3B) via Ollama, ChromaDB & Chat History...")
    print("=" * 60)

    try:
        agent = RAGAgent(model_name="qwen2.5:3b", k_results=3)
        session_id = "user_session_01"
        
        print("\nSystem ready! You can now search for dishes and chat interactively.")
        print("Type 'exit', 'quit', or press Ctrl+C to stop.\n")

        while True:
            query = input("User Query: ").strip()
            
            if not query:
                continue
            if query.lower() in ["exit", "quit"]:
                print("\nExiting Smart Menu Assistant. Goodbye!")
                break

            print("\nAnalyzing context, constraints, and generating recommendation...")
            response = agent.ask(query, session_id=session_id)
            
            print("-" * 60)
            print(f"🤖 Assistant Response:\n{response}")
            print("-" * 60 + "\n")

    except Exception as e:
        print(f"\nError during execution: {e}")
        print("Tip: Make sure Ollama is running in the background (`ollama serve`).")

if __name__ == "__main__":
    main()