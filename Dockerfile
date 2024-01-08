# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Install OpenGL libraries
RUN apt-get update \     
     && apt-get install -y libgl1-mesa-glx \
     && apt-get install -y libglib2.0-0

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that Gunicorn will run on
EXPOSE 5000

# Command to run your application with Gunicorn
CMD ["python3","server.py"]
