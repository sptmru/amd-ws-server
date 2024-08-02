FROM python:3.6.15-bullseye

RUN apt-get update && apt-get install -y \
    make automake python3-dev gcc libffi7 libssl-dev build-essential libssl-dev \
    g++ subversion zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

RUN pip install --upgrade pip wheel setuptools

ENV PYTHONUNBUFFERED 1

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install mkl
RUN ldconfig

COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

COPY . .
COPY .env.template .env

ENTRYPOINT ["docker-entrypoint.sh"]
