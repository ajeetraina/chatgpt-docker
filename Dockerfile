FROM python:3.8

# Install necessary dependencies
RUN pip install transformers==3.3.0

# Copy the ChatGPT model code and any necessary configuration files
COPY . /app
WORKDIR /app

# Run the ChatGPT model code
CMD ["python", "chat_gpt.py"]
