# BugInsight Swarm 🐝

**Microsoft Build AI 2026 Hackathon Submission**
**Theme:** AI-Powered Production Function

BugInsight Swarm is an autonomous, multi-agent AI system designed to eliminate the engineering bottleneck of triaging and fixing repository-wide bugs. By ingesting a GitHub issue, BugInsight dynamically clones the repository, semantically indexes its contents, predicts the severity of the bug, isolates the root cause, and generates a PR-ready patch—all orchestrated through a real-time event stream.

---

## 🏗️ Architecture

BugInsight leverages an 11-agent LangGraph workflow backed by specialized models and automated verification logic.

```mermaid
graph TD
    A[User / Next.js Dashboard] -->|Submit GitHub URL & Issue| B(FastAPI Backend)
    B --> C{LangGraph Swarm}
    
    subgraph LangGraph Orchestration
        C --> D[1. Repository Agent]
        D --> S[2. Scanner Agent]
        S --> E[3. Severity Agent]
        E --> F[4. Root Cause Agent]
        F --> G[5. Fix Agent]
        G --> V[6. Validation Agent]
        V --> T[7. Test Agent]
        T --> X[8. Test Execute Agent]
        X --> R[9. Auto Rescan Agent]
        R --> H[10. GitHub Agent]
        H --> I[11. Sprint Agent]
    end
    
    D <--> J[(ChromaDB Cache)]
    D --> K[Git Clone & Chunking]
    
    S <--> Sem[Semgrep & Bandit]
    
    E <--> L((Fine-Tuned CodeBERT))
    
    F <--> M((Ollama Qwen2.5-Coder))
    G <--> M
    I <--> M
    
    H <--> N[GitHub API / Mock Mode]
    
    C -->|Server-Sent Events Stream| A
```

### The Swarm Agents
1. **Repository Agent:** Performs a `git clone`, chunks source code, and semantically indexes the entire repository into ChromaDB. Built with a high-performance **Repository Cache**.
2. **Scanner Agent:** Runs Semgrep and Bandit against the cloned repository to find specific vulnerable code blocks based on the user's issue.
3. **Severity Agent:** Interfaces with a custom-trained **CodeBERT** model to predict the critical severity (P0-P4) of the issue.
4. **Root Cause Agent:** Uses **Qwen2.5-Coder** via local Ollama to ingest the semantic search context and deduce the exact file and lines causing the bug.
5. **Fix Agent:** Generates a full-file, context-aware replacement fix.
6. **Validation Agent:** Provides strict Functional Preservation checks to ensure the fix safely patches the code without destroying surrounding logic or introducing secondary vulnerabilities.
7. **Test Agent:** Generates Pytest regression tests tailored specifically for the generated patch.
8. **Test Execute Agent:** Safely executes the generated regression tests against the patched code inside an isolated workspace.
9. **Auto Rescan Agent:** Re-runs the scanner tools on the patched code to verify the original vulnerability has been eradicated.
10. **GitHub Agent:** Opens a live Pull Request containing the patch, validation score, test results, and rescan status via the GitHub API (or operates in mock mode).
11. **Sprint Agent:** Calculates estimated developer time saved and assigns Agile story points.

## 🚀 Key Features

* **Dynamic Repository Indexing:** Point BugInsight at *any* public repository. It clones it on the fly and builds a semantic understanding of the codebase.
* **Intelligent Caching:** Hashing algorithms ensure that once a repository is cloned and embedded into ChromaDB, subsequent analyses are near-instantaneous.
* **Real-time SSE Dashboard:** A sleek Next.js frontend built with React Strict Mode safety to stream agent completions, patch generation, and CodeBERT predictions in real-time.
* **100% Local Inferencing Capability:** Capable of running entirely on local hardware (Ollama + local PyTorch models) for maximum security and privacy.

## 💻 Tech Stack

* **Frontend:** Next.js (App Router), React, Tailwind CSS
* **Backend:** FastAPI, Server-Sent Events (SSE), Uvicorn
* **Orchestration:** LangGraph
* **AI/ML:** PyTorch, HuggingFace (CodeBERT), Ollama (Qwen2.5-Coder:7b)
* **Vector Store:** ChromaDB (SentenceTransformers)

## 🏁 Getting Started

### Prerequisites
* Python 3.10+
* Node.js 18+
* Ollama installed and running (`qwen2.5-coder:7b` pulled)

### 1. Start the Backend
```bash
# Start Ollama (if not already running as a service)
ollama serve

# Install python dependencies
pip install -r requirements.txt

# Start the FastAPI server
python -m uvicorn swarm.api:app --host 0.0.0.0 --port 8000
```

### 2. Start the Frontend
```bash
cd frontend
npm install
npm run dev
```

Navigate to `http://localhost:3000` to launch the Swarm.

---
*Built with ❤️ for Microsoft Build AI 2026*
