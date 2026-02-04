import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from app.config import settings
from app.models import LLMManager
from app.services import SessionManager, ConversationPersistence
from app.routers import chat_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - load model at startup, cleanup at shutdown."""
    # Startup
    logger.info("Starting LLM Feedback Chat application...")

    # Initialize services
    app.state.llm_manager = LLMManager()
    app.state.session_manager = SessionManager()
    app.state.persistence = ConversationPersistence()

    # Load the model (this may take a while)
    logger.info(f"Loading model: {settings.model_name}")
    app.state.llm_manager.load_model()
    logger.info("Model loaded successfully!")

    yield

    # Shutdown
    logger.info("Shutting down...")
    app.state.llm_manager.unload_model()
    app.state.persistence.cleanup_locks()
    logger.info("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    lifespan=lifespan,
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Include API router
app.include_router(chat_router, prefix="/api")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main chat interface."""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "app_name": settings.app_name,
            "model_name": settings.model_name,
        },
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": app.state.llm_manager.is_loaded,
        "model_name": app.state.llm_manager.model_name,
    }
