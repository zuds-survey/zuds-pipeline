FROM ubuntu:17.10

RUN apt-get update && \
    apt-get install -y curl build-essential software-properties-common && \
    curl -sL https://deb.nodesource.com/setup_8.x | bash - && \
    apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y python3.6 python3.6-venv python3.6-dev \
                       libpq-dev supervisor openssh-server \
                       git nginx nodejs postgresql-client && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    useradd --create-home --shell /bin/bash skyportal

RUN python3.6 -m venv /skyportal_env && \
    \
    bash -c "source /skyportal_env/bin/activate && \
    pip install --upgrade pip"

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ADD . /skyportal
WORKDIR /skyportal

RUN bash -c "\
    source /skyportal_env/bin/activate && \
    \
    make -C baselayer paths && \
    (make -f baselayer/Makefile baselayer dependencies || make -C baselayer dependencies)"

RUN bash -c "\
    \
    (make -f baselayer/Makefile bundle || make -c baselayer bundle) && \
    \
    chown -R skyportal.skyportal /skyportal_env && \
    chown -R skyportal.skyportal /skyportal && \
    \
    cp docker.yaml config.yaml"

RUN mkdir -p static/thumbnails /var/run/sshd /run/sshd /root/.ssh && \
    echo 'root:root' |chpasswd && \
    sed -ri 's/^#?PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config

RUN bash -c "\
    source /skyportal_env/bin/activate && \
    cd / && git clone https://github.com/dannygoldstein/sncosmo.git && \
    cd sncosmo && python setup.py install && cd - && rm -rf sncosmo"

EXPOSE 5000 5001 22

CMD bash -c "source /skyportal_env/bin/activate && \
             (make log &) && \
             make run_production"
