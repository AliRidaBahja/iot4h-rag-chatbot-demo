# Use official Python image as base
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy everything from your streamlit_app directory into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit's default port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
