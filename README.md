# datagen
David J. Lee, 2022-2024

![L-systems](https://github.com/djslzx/datagen/blob/5902000661da4cf74e12a4acc147040608632ce5/examples/lsys/ns/sample1.png)


Synthesize novel and diverse datasets for low-resource program synthesis domains.

## Domains
- Lindenmayer systems
- Mujoco/2D ant walker programs
- Python programming puzzles
- Regular expressions

## Samples
### Lindenmayer systems
![Sample 1](https://github.com/djslzx/datagen/blob/5902000661da4cf74e12a4acc147040608632ce5/examples/lsys/ns/sample2.png)
![Sample 2](https://github.com/djslzx/datagen/blob/5902000661da4cf74e12a4acc147040608632ce5/examples/lsys/ns/sample3.png)
![Sample 3](https://github.com/djslzx/datagen/blob/5902000661da4cf74e12a4acc147040608632ce5/examples/lsys/ns/sample4.png)
![Sample 4](https://github.com/djslzx/datagen/blob/5902000661da4cf74e12a4acc147040608632ce5/examples/lsys/ns/sample5.png)

### Ant walker paths
![Ant outputs!](https://github.com/djslzx/datagen/blob/5902000661da4cf74e12a4acc147040608632ce5/examples/ant/oriented-trails.png)

### Programming problems
See `examples/puzzles`

## Framework
Treat dataset generation as Markov chain Monte Carlo search over sets of candidate programs, with the following proposal distributions:
- LLM: generate new programs via prompting an LLM with program mutation prompts (Python puzzles)
- PCFG: fit a probabilistic context-free grammar (PCFG) to a program or set of programs using inside-outside, and then sample from the fitted grammar
 
and the following target distributions:

- energy: maximize a physically motivated "energy" function over program embeddings (pulled from an off-the-shelf embedding model and compressed using dimensionality reduction techniques when necessary)
- variance: maximize the sum of embedding distances from the mean embedding
among other variations

Note: this is research code, so browse at your own peril. :)
