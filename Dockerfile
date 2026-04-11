FROM python:3.11-slim

WORKDIR /app

# system requirements (to fasten sklearn/catboost)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# copy dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# default command for a docker to run
CMD ["bash"]