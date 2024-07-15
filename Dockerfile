# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements.txt first to leverage Docker cache
COPY requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the application code
COPY . /app

# Make port 80 available to the world outside this container
EXPOSE 80

# Run the script when the container launches
CMD ["python", "examples/generate_fuzz.py"]
