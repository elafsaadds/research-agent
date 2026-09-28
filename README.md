# Research Agent

A local Research Agent project built with Python, VS Code, Ollama, and a local Large Language Model (LLM).

## Project Progress

The following components have been successfully implemented:

* ✅ Python environment
* ✅ VS Code project
* ✅ Ollama
* ✅ Local LLM
* ✅ Qwen 2.5 1.5B
* ✅ `model.py`
* ✅ `get_model()`

## Current Architecture

```text
VS Code Project
      │
      ▼
Python Environment (.venv)
      │
      ▼
research_agent
      │
      ▼
model.py
      │
      ▼
get_model()
      │
      ▼
Ollama
      │
      ▼
Qwen 2.5 1.5B
      │
      ▼
Local LLM Response
```

## Technologies Used

| Technology    | Purpose                       |
| ------------- | ----------------------------- |
| Python        | Main programming language     |
| VS Code       | Development environment       |
| Ollama        | Local LLM runtime             |
| Qwen 2.5 1.5B | Local language model          |
| Requests      | Communication with Ollama API |
| Pytest        | Testing                       |

## Project Structure

```text
research-agent/
│
├── src/
│   └── research_agent/
│       ├── __init__.py
│       ├── model.py
│       └── baseline.py
│
├── tests/
│
├── .venv/
│
└── README.md
```

## Python Environment

The project uses a Python virtual environment:

```text
.venv
```

The environment is activated before running the project.

Example:

```powershell
.\.venv\Scripts\Activate.ps1
```

After activation, the terminal should show:

```text
(research-agent) PS C:\Users\Dell\projects\research-agent>
```

## Ollama Setup

Ollama is used to run the language model locally.

The installed model is:

```text
qwen2.5:1.5b
```

To check installed models:

```powershell
ollama list
```

To run the model directly:

```powershell
ollama run qwen2.5:1.5b
```

## Model Configuration

The project uses environment variables to select the model provider and model name.

```powershell
$env:MODEL_PROVIDER = "ollama"
$env:OLLAMA_MODEL = "qwen2.5:1.5b"
```

The Ollama API runs locally at:

```text
http://localhost:11434
```

## `model.py`

The `model.py` module provides the interface between the Research Agent and Ollama.

The main function is:

```python
get_model()
```

It creates the local Ollama model interface.

Example:

```python
from research_agent.model import get_model

model = get_model()
```

The model can then generate a response:

```python
response = model.generate([
    {
        "role": "user",
        "content": "Say hi in 3 words"
    }
])

print(response.text)
```

## Model Response

The response object provides:

* Generated text
* Response latency
* Completion token count

Example:

```text
Hello there! | 2.69 s | 4 tokens
```

This confirms that the local LLM is successfully responding through the Python application.

## Verification

The model integration was tested successfully using:

```powershell
python -c "from research_agent.model import get_model; r = get_model().generate([{'role':'user','content':'Say hi in 3 words'}]); print(r.text, '|', r.latency_s, 's', '|', r.completion_tokens, 'tokens')"
```

Successful output:

```text
Hello there! | 2.693581000001359 s | 4 tokens
```

## Testing

The project uses `pytest` for testing.

Current test result:

```text
1 passed
```

Run the tests with:

```powershell
pytest -q
```

## Git and GitHub

The project is version-controlled using Git and hosted on GitHub.

Repository:

```text
https://github.com/elafsaadds/research-agent
```

Current branch:

```text
main
```

The latest implementation was committed and pushed to GitHub.

Latest commit:

```text
806ef4a
Add Ollama model integration and baseline agent
```

## Current Status

### Completed

* [x] Python environment
* [x] VS Code project
* [x] Ollama installation
* [x] Local LLM setup
* [x] Qwen 2.5 1.5B
* [x] Ollama-Python integration
* [x] `model.py`
* [x] `get_model()`
* [x] Baseline agent
* [x] Basic testing
* [x] GitHub integration
