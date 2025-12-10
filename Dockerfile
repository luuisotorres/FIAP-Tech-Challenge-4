FROM python:3.12-slim-bookworm

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy dependency files first
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-install-project

# Copy the rest of the application
COPY . .

# Install the project itself
RUN uv sync --frozen

# Expose the port
EXPOSE 8000

# Run the application
CMD ["uv", "run", "uvicorn", "fiap_tech_challenge_4.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "src"]
