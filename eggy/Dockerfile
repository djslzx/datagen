FROM messense/manylinux2014-cross:x86_64
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH=$PATH:~/.cargo/bin
RUN pip install maturin