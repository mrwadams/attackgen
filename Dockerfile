# Use specific version with SHA256 for reproducibility and security
FROM python:3.12-slim@sha256:d86b4c74b936c438cd4cc3a9f7256b9a7c27ad68c7caf8c205e18d9845af0164

# Create non-root user for security
RUN groupadd -r attackgen && \
    useradd -r -g attackgen -u 1000 -m -d /home/attackgen attackgen

# Set working directory
WORKDIR /app

# Copy and install dependencies as root (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip check

# Set up Streamlit cache directory with proper permissions
RUN mkdir -p /home/attackgen/.streamlit && \
    chown -R attackgen:attackgen /home/attackgen

# Copy application files with proper ownership
COPY --chown=attackgen:attackgen . .

# Switch to non-root user
USER attackgen

# Expose port
EXPOSE 8501

# Health check using Python urllib instead of curl
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=2)" || exit 1

# Run application with security enhancements
ENTRYPOINT ["streamlit", "run", "00_👋_Welcome.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.enableXsrfProtection=true"]
