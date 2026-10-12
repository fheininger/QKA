# Reproduction environment for the QKA results, written for Code Ocean.
#
# Code Ocean only accepts its own base images, pulled through the REGISTRY_HOST
# build argument it sets itself. Use this file as environment/Dockerfile of
# the capsule. The base image ships Python 3.12, so the environment the
# results were produced in (Python 3.11.13, pins identical to requirements.txt
# and environment.yml) is created as a conda environment on top of it and put
# first on PATH. Every package comes from a prebuilt wheel.
#
# Plain Docker works too, with the registry given explicitly:
#   docker build --build-arg REGISTRY_HOST=registry.codeocean.com -t qka .
#   docker run --rm -v "$PWD:/code" -v "$PWD/results:/results" qka ./run
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN mamba create -y -n qka -c conda-forge --override-channels python=3.11.13 pip \
    && mamba clean -y --all

ENV PATH=/opt/conda/envs/qka/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    QSVM_RESULTS=/results

RUN pip install --no-cache-dir --only-binary=:all: \
        numpy==1.26.4 \
        scipy==1.11.4 \
        pandas==2.3.1 \
        scikit-learn==1.8.0 \
        imbalanced-learn==0.14.1 \
        pennylane==0.37.0 \
        pennylane-lightning==0.37.0 \
        autoray==0.7.2 \
        autograd==1.8.0 \
        networkx==2.8.8 \
        rustworkx==0.16.0 \
        matplotlib==3.10.5 \
        seaborn==0.13.2 \
        joblib==1.5.1 \
        threadpoolctl==3.6.0 \
        tqdm==4.67.1
