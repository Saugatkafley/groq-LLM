FROM python:3.10-slim

# Add a non-root user and switch to it
RUN useradd -m -u 1000 user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy the current directory contents into the container at $HOME/app
COPY --chown=user . $HOME/app

# Install dependencies including gcc and graphviz
USER root
RUN apt-get update && \
    apt-get install -y gcc graphviz graphviz-dev libgraphviz-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python packages as non-root user
USER user
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"
ENV GROQ_API_KEY=$(cat /run/secrets/GROQ_API_KEY)

CMD ["gradio", "app.py"]
