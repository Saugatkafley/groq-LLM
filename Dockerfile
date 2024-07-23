FROM python:3.10-slim
WORKDIR /user/src/app
COPY . .

# Install dependencies including gcc and graphviz
RUN apt-get update && \
    apt-get install -y gcc graphviz graphviz-dev libgraphviz-dev 
# apt-get clean && rm -rf /var/lib/apt/lists/*
# Update pip to the latest version

RUN pip install --upgrade pip

# Install pygraphviz with --build-option flags
# RUN pip install --no-cache-dir --build-option="--include-path=/usr/include/graphviz" --build-option="--library-path=/usr/lib/graphviz" pygraphviz

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Expose port
EXPOSE 6333
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"
CMD [ "gradio","app.py" ]