# Use specific version with SHA256 for reproducibility and security
FROM python:3.14-slim@sha256:9aaa474097d094040649f2162a627c0049b408f52138d980916788d80f04802f

# The base image ships zlib1g 1:1.3.dfsg+really1.3.1-1+b1, which carries
# CVE-2026-85091, CVE-2026-76844 and CVE-2026-27171. trixie-security has the
# fix (+deb13u1). Drop this once the pinned digest includes it.
RUN apt-get update && \
    apt-get install -y --no-install-recommends --only-upgrade zlib1g && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r attackgen && \
    useradd -r -g attackgen -u 1000 -m -d /home/attackgen attackgen

# Set working directory
WORKDIR /app

# Copy application files with proper ownership
COPY --chown=attackgen:attackgen . .

# Install dependencies as root. requirements.txt is `-e .`, which needs the
# package sources above in place, so this layer follows the full copy.
# Upgrade pip first to pick up fixes for CVE-2025-8869 (symlink extraction)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip check

# Set up Streamlit cache directory with proper permissions
RUN mkdir -p /home/attackgen/.streamlit && \
    chown -R attackgen:attackgen /home/attackgen

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
