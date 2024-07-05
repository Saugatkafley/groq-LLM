FROM python:3.10-slim

WORKDIR /user/src/app
COPY . .

# Install dependencies including gcc and graphviz
RUN apt-get update && \
    apt-get install -y gcc graphviz graphviz-dev libgraphviz-dev

# apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"
CMD [ "gradio" ,"app.py" ]
# CMD ["python", "/user/src/app/app.py"]
