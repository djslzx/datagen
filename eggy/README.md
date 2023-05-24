# eggy
Using `egg` to simplify programs

## Cross-compiling for G2 cluster
To compile to a manylinux-compatible target from macOS, we use one of these Docker images: https://github.com/rust-cross/manylinux-cross.
To make this easier, we build a Docker image that extends manylinux-cross to add rust and install maturin. 

Build the Docker image:
```shell
$ docker build -t maturin:latest 
```
Run:
```shell
$ docker run -it --platform linux/amd64 -v \ 
   ~/Documents/research/prob-repl/eggy:/home/shared \
   messense/manylinux2014-cross:x86_64
```
Building on the Docker image directly won't work because of macOS and cargo shenanigans:
https://github.com/rust-lang/cargo/issues/10781

So use this instead:
```shell
$ cargo build --config net.git-fetch-with-cli=true
```
Then run `maturin build --release`.  And we're done!