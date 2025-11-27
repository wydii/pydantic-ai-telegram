"""
Local Whisper transcription service using openai-whisper.

Based on: https://github.com/openai/whisper
"""

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Any, Optional

from pydantic_ai_telegram.transcription.base import TranscriptionService

logger = logging.getLogger(__name__)


def check_ffmpeg_installed() -> bool:
    """
    Check if ffmpeg is installed on the system.
    
    Returns:
        True if ffmpeg is available, False otherwise
    """
    return shutil.which("ffmpeg") is not None


def get_ffmpeg_install_instructions() -> str:
    """
    Get platform-specific ffmpeg installation instructions.
    
    Returns:
        Installation instructions string
    """
    import platform
    
    system = platform.system().lower()
    
    if system == "darwin":  # macOS
        return "Install with: brew install ffmpeg"
    elif system == "linux":
        # Try to detect distro
        try:
            with open("/etc/os-release") as f:
                os_info = f.read().lower()
                if "ubuntu" in os_info or "debian" in os_info:
                    return "Install with: sudo apt update && sudo apt install ffmpeg"
                elif "arch" in os_info:
                    return "Install with: sudo pacman -S ffmpeg"
                elif "fedora" in os_info or "rhel" in os_info:
                    return "Install with: sudo dnf install ffmpeg"
        except Exception:
            pass
        return "Install ffmpeg using your package manager"
    elif system == "windows":
        return "Install with: choco install ffmpeg (Chocolatey) or scoop install ffmpeg (Scoop)"
    
    return "Install ffmpeg from: https://ffmpeg.org/download.html"


class LocalWhisperTranscription(TranscriptionService):
    """
    Transcription service using local Whisper model.
    
    Requires:
    - openai-whisper package: pip install openai-whisper
    - ffmpeg installed on system
    
    Available models (from fastest to most accurate):
    - tiny: 39M params, ~1GB RAM, ~10x speed
    - base: 74M params, ~1GB RAM, ~7x speed
    - small: 244M params, ~2GB RAM, ~4x speed
    - medium: 769M params, ~5GB RAM, ~2x speed
    - large: 1550M params, ~10GB RAM, 1x speed
    - turbo: 809M params, ~6GB RAM, ~8x speed (recommended)
    
    Reference: https://github.com/openai/whisper
    """
    
    def __init__(
        self,
        model_name: str = "turbo",
        device: Optional[str] = None,
        language: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        """
        Initialize local Whisper transcription service.
        
        Args:
            model_name: Whisper model size
                - "tiny" - Fastest, least accurate
                - "base" - Fast, decent accuracy
                - "small" - Good balance
                - "medium" - High quality
                - "large" - Best quality, slowest
                - "turbo" - Optimized large model (recommended)
            device: Device to use
                - None: Auto-detect (CUDA if available, else CPU)
                - "cpu": Force CPU
                - "cuda": Force CUDA/GPU
            language: ISO 639-1 language code for transcription
                - None: Auto-detect language (multilingual)
                - "en": English
                - "fr": French
                - "es": Spanish
                - etc. (see Whisper docs for full list)
            verbose: Show detailed transcription progress
            
        Raises:
            ImportError: If openai-whisper is not installed
            RuntimeError: If ffmpeg is not installed
        """
        self.model_name = model_name
        self.device = device
        self.language = language
        self.verbose = verbose
        self.model: Optional[Any] = None
        
        # Check for openai-whisper
        try:
            import whisper
            self.whisper = whisper
        except ImportError:
            raise ImportError(
                "openai-whisper is not installed.\n"
                "Install it with: pip install openai-whisper\n"
                "Or: pip install pydantic-ai-telegram[whisper]"
            )
        
        # Check for ffmpeg
        if not check_ffmpeg_installed():
            instructions = get_ffmpeg_install_instructions()
            raise RuntimeError(
                f"ffmpeg is not installed on your system.\n"
                f"{instructions}\n\n"
                f"ffmpeg is required for audio processing with Whisper."
            )
        
        logger.info(f"Initialized LocalWhisperTranscription with model={model_name}, language={language or 'auto-detect'}")
    
    def _load_model(self) -> Any:
        """
        Load the Whisper model (lazy loading).
        
        Returns:
            Loaded Whisper model
        """
        if self.model is None:
            logger.info(f"Loading Whisper model '{self.model_name}'...")
            logger.info("First time may take a while to download the model")
            
            try:
                self.model = self.whisper.load_model(self.model_name, device=self.device)
                logger.info(f"✓ Whisper model '{self.model_name}' loaded successfully")
                
                # Log device info
                if hasattr(self.model, 'device'):
                    logger.info(f"  Running on: {self.model.device}")
                
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                raise
        
        return self.model
    
    async def transcribe(self, audio_file_path: str | Path) -> str:
        """
        Transcribe an audio file using local Whisper model.
        
        This method:
        1. Loads the model if not already loaded
        2. Processes the audio file
        3. Returns the transcribed text
        
        Args:
            audio_file_path: Path to the audio file
            
        Returns:
            Transcribed text
            
        Raises:
            FileNotFoundError: If audio file doesn't exist
            Exception: If transcription fails
        """
        audio_path = Path(audio_file_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Load model if not already loaded
        model = self._load_model()
        
        # Run transcription in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._transcribe_sync,
            model,
            str(audio_path),
        )
        
        return result["text"].strip()
    
    def _transcribe_sync(self, model: Any, audio_path: str) -> dict:
        """
        Synchronous transcription (runs in thread pool).
        
        Args:
            model: Loaded Whisper model
            audio_path: Path to audio file
            
        Returns:
            Transcription result dictionary with keys:
            - text: Transcribed text
            - segments: Detailed segments with timestamps
            - language: Detected language
        """
        # Build transcription options
        options: dict[str, Any] = {
            "verbose": self.verbose,
        }
        
        # Add language if specified
        if self.language:
            options["language"] = self.language
            logger.info(f"Transcribing audio (language: {self.language}): {audio_path}")
        else:
            logger.info(f"Transcribing audio (auto-detect language): {audio_path}")
        
        try:
            # Transcribe using Whisper
            result = model.transcribe(audio_path, **options)
            
            # Log detected language if auto-detect was used
            if not self.language and "language" in result:
                detected_lang = result["language"]
                logger.info(f"  Detected language: {detected_lang}")
            
            logger.info(f"  Transcription completed ({len(result['text'])} characters)")
            
            return result
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    async def close(self) -> None:
        """
        Clean up resources (unload model).
        
        Whisper models don't have explicit cleanup methods,
        but we clear the reference to allow garbage collection.
        """
        if self.model is not None:
            self.model = None
            logger.info("Whisper model unloaded")
    
    def get_available_models(self) -> list[str]:
        """
        Get list of available Whisper models.
        
        Returns:
            List of model names
        """
        return ["tiny", "base", "small", "medium", "large", "turbo"]
    
    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        model_info = {
            "tiny": {"params": "39M", "ram": "~1GB", "speed": "~10x"},
            "base": {"params": "74M", "ram": "~1GB", "speed": "~7x"},
            "small": {"params": "244M", "ram": "~2GB", "speed": "~4x"},
            "medium": {"params": "769M", "ram": "~5GB", "speed": "~2x"},
            "large": {"params": "1550M", "ram": "~10GB", "speed": "1x"},
            "turbo": {"params": "809M", "ram": "~6GB", "speed": "~8x"},
        }
        
        return {
            "model": self.model_name,
            "language": self.language or "auto-detect",
            "device": self.device or "auto",
            "info": model_info.get(self.model_name, {}),
            "ffmpeg_available": check_ffmpeg_installed(),
        }

