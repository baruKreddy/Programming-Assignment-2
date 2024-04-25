# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install Java (required for PySpark)
RUN apt-get update && apt-get install -y openjdk-11-jre-headless && \
    apt-get clean;

# Install Python packages
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable for Java (this might be needed depending on your setup)
ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64

# Run app.py when the container launches
CMD ["python", "predict.py"]

