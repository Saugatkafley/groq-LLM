FROM python:3.10-slim
RUN useradd -m -u 1000 user
# switch to user
USER user
# Set home to the user's home directory
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR /$HOME/app


# Install dependencies including gcc and graphviz
RUN apt-get update && \
    apt-get install -y gcc graphviz graphviz-dev libgraphviz-dev \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Try and run pip command after setting the user with `USER user` to avoid permission issues with Python
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user . $HOME/app

# Expose port
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"
ENV GROQ_API_KEY=$(cat /run/secrets/GROQ_API_KEY)
CMD [ "gradio" ,"app.py" ]
# CMD ["python", "/user/src/app/app.py"]
