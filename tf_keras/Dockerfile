FROM tensorflow/tensorflow:devel-gpu

LABEL maintainer="Mehran Ghandehari <mehran.maps@gmail.com>"

RUN apt-get update

WORKDIR /tmp/

COPY . .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /tmp/requirements.txt

CMD "ls"