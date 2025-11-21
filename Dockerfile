# Base image with Python and uv preinstalled
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY datasets ./datasets
COPY analysis ./analysis
COPY main.py ./

# Install dependencies using uv
RUN uv sync --frozen

# Default environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=production

# Expose port
EXPOSE 8501

# Run the application
CMD ["uv", "run", "python", "-m", "app", "--server.port=8501", "--server.address=0.0.0.0"]
