# CloverDX LLM Chat

A multi-user web application for testing and evaluating LLM models with conversation logging and feedback collection.

![CloverDX LLM Chat](static/images/cloverdx-logo-white.svg)

## Features

- **Web-based Chat Interface** - Clean, responsive UI for interacting with LLM models
- **Multi-user Support** - Session-based isolation allows multiple concurrent users
- **Conversation Persistence** - All conversations saved as JSON files organized by date
- **Feedback Collection** - Rate responses (1-5 stars), add comments, suggest better responses
- **Markdown Rendering** - LLM responses rendered with full Markdown support and syntax highlighting
- **Flexible Configuration** - Configure model, GPU devices, generation parameters via `.env` file
- **GPU Support** - CUDA device selection, quantization (4-bit/8-bit), multi-GPU support

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/dpavlis/llm_feedback.git
cd llm_feedback

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure

```bash
# Copy example configuration
cp .env.example .env

# Edit .env with your settings (model, GPU, etc.)
```

### 3. Run

```bash
python run.py
```

Open http://localhost:8000 in your browser.

## Configuration

All settings can be configured in `.env` file. Key options:

| Setting | Default | Description |
|---------|---------|-------------|
| `PORT` | 8000 | HTTP server port |
| `MODEL_NAME` | Qwen/Qwen2.5-Coder-7B-Instruct | HuggingFace model ID |
| `MODEL_PATH` | - | Local path to model (overrides MODEL_NAME) |
| `MODEL_DEVICE` | auto | Device: `cuda`, `cuda:0`, `mps`, `cpu` |
| `CUDA_VISIBLE_DEVICES` | - | Restrict visible GPUs (e.g., `0,1`) |
| `LOAD_IN_4BIT` | false | Enable 4-bit quantization |
| `LOAD_IN_8BIT` | false | Enable 8-bit quantization |
| `TEMPERATURE` | 0.7 | Sampling temperature |
| `MAX_RESPONSE_TOKENS` | 1024 | Max tokens per response |
| `SYSTEM_PROMPT` | - | Optional system prompt for all conversations |

See `.env.example` for all available options.

## Project Structure

```
llm_feedback/
├── app/
│   ├── config.py           # Configuration (pydantic-settings)
│   ├── main.py             # FastAPI application
│   ├── models/
│   │   └── llm_manager.py  # Model loading & inference
│   ├── routers/
│   │   └── chat.py         # API endpoints
│   ├── schemas/
│   │   └── chat.py         # Pydantic models
│   └── services/
│       ├── persistence.py  # JSON file storage
│       └── session.py      # Session management
├── static/
│   ├── css/style.css       # UI styling
│   └── js/chat.js          # Frontend logic
├── templates/
│   └── index.html          # Chat interface
├── data/conversations/     # Stored conversations (by date)
├── .env.example            # Configuration template
├── requirements.txt        # Python dependencies
└── run.py                  # Startup script
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Chat interface |
| `/api/conversations` | POST | Create new conversation |
| `/api/conversations` | GET | List conversations |
| `/api/conversations/{id}` | GET | Get conversation details |
| `/api/conversations/{id}` | DELETE | Remove from session |
| `/api/chat` | POST | Send message, get response |
| `/api/conversations/{id}/feedback` | POST | Submit feedback |
| `/health` | GET | Health check |

## Conversation Storage

Conversations are saved as JSON files organized by date:

```
data/conversations/
└── 2024/
    └── 02/
        └── 04/
            └── conv_abc123.json
```

Each file contains:
- Conversation metadata (model, user name, timestamps)
- Full message history
- Feedback for each response

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA-capable GPU (recommended) or Apple Silicon (MPS) or CPU

## GPU Memory Requirements

| Model Size | FP16 | 8-bit | 4-bit |
|------------|------|-------|-------|
| 7B | ~14GB | ~7GB | ~4GB |
| 13B | ~26GB | ~13GB | ~7GB |

## License

MIT License
