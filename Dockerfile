FROM python:3.10-slim
# Set home to the user's home directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /$HOME/app

# Install dependencies including gcc and graphviz
RUN apt-get update && \
    apt-get install -y gcc graphviz graphviz-dev libgraphviz-dev \
# apt-get clean && rm -rf /var/lib/apt/lists/*

# Update pip to the latest version
RUN pip install --upgrade pip

# Install pygraphviz with --build-option flags
RUN pip install --no-cache-dir --build-option="--include-path=/usr/include/graphviz" --build-option="--library-path=/usr/lib/graphviz" pygraphviz

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user . $HOME/app

# Expose port
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"
RUN --mount=type=secret,id=GROQ_API_KEY,mode=0444,required=true
ENV GROQ_API_KEY=GROQ_API_KEY
CMD [ "gradio","app.py" ]