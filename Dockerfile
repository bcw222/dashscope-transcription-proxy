FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml LICENSE ./
COPY src/ ./src/

RUN pip install --no-cache-dir .

ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=info
ENV HOST=0.0.0.0
ENV PORT=8000

# Run as non-root
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000

CMD ["python", "-m", "dashscope_transcription_proxy"]
