# Use the official Python base image with version tag
FROM python:3.11-slim

# Set the working directory
ADD . /app
WORKDIR /app

#RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list
#
## Update package lists and install build-essential
#RUN apt-get update && apt-get install -y build-essential

# Copy files to the container
COPY . .

RUN pip3 install --upgrade pip

# Install the project dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port on which the FastAPI application runs
EXPOSE 8000

# Add the necessary headers to enable CORS
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/app"

# Copy the script that runs the update_vectorstore() function
COPY vectorstore_update.sh /app

# Run the update_vectorstore() function before starting the FastAPI application
CMD ["bash", "vectorstore_update.sh"]