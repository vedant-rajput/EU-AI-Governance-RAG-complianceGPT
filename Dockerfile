FROM python:3.12-slim

WORKDIR /app

# Install system dependencies needed for PyMuPDF or others
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose ports for Gradio and FastMCP
EXPOSE 7860 8001

# Run the Gradio app
CMD ["python", "app/app.py"]
