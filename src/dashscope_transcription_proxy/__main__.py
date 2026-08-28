"""Entry-point for ``python -m dashscope_transcription_proxy``."""

import os
import uvicorn

def main() -> None:
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(
        "dashscope_transcription_proxy.app:app",
        host=host,
        port=port,
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
    )

if __name__ == "__main__":
    main()
