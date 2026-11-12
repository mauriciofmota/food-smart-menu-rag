# Food Smart Menu & Intent Assistant (Local RAG)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/Ollama-Qwen%202.5-orange.svg)](https://ollama.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-green.svg)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A privacy-first, local **Retrieval-Augmented Generation (RAG)** prototype designed to solve the "choice paralysis" in food delivery platforms by processing natural language intent and matching it semantically against restaurant menus.

---

## Product Vision & Business Problem

### The Challenge (Choice Overload)
When users open a food delivery app like iFood, they often experience **choice paralysis** — browsing through hundreds of restaurants without knowing exactly what they want. Traditional keyword searches fail when users search by *moment, mood, or complex dietary constraints* (e.g., *"I need a high-protein, lactose-free meal under $40 that arrives in 30 minutes"*).

### The Solution
This project simulates an AI-powered conversational intent filter that sits between the user's search query and the catalog database. It uses local open-source LLMs and vector embeddings to extract strict constraints and deliver hyper-personalized, context-aware menu recommendations instantly and securely.

---

## System Architecture & Workflow

The pipeline runs 100% locally on standard consumer hardware, ensuring zero data leakage and minimal operational costs.

> **⚠️ Disclaimer:** Este é um **projeto pessoal de portfólio e estudos**, sem fins lucrativos ou vínculo comercial com qualquer empresa de delivery. O objetivo do repositório é exclusivamente validar habilidades técnicas em Inteligência Artificial Generativa, Engenharia de Dados e arquiteturas RAG (Retrieval-Augmented Generation).
