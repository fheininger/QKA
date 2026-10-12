# Reproduction environment for the QKA results, for Code Ocean or plain Docker.
#
# Standalone use (the repo is mounted, not copied, so results land next to it):
#   docker build -t qka .
#   docker run --rm -v "$PWD:/code" -v "$PWD/results:/results" qka
#   docker run --rm -v "$PWD:/code" qka python verify_fast_kernel.py
#
# Code Ocean: use as environment/Dockerfile. The capsule mounts the code at
# /code and expects output under /results, which QSVM_RESULTS points at.
#
# Same pins as requirements.txt and environment.yml: Python 3.11.13 on Linux
# x86_64, every package from a prebuilt wheel (no compiler in the image).
# PennyLane 0.37 needs numpy < 2, and autoray/autograd/networkx/rustworkx are
# pinned because PennyLane does not upper-bound them and newer releases break
# `import pennylane`. Keep the three files in sync.
FROM python:3.11.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
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
        tqdm==4.67.1 \
    && mkdir -p /code /results

WORKDIR /code
CMD ["python", "QSVM_template.py"]
