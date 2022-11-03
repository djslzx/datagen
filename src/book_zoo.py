from typing import List, Tuple
from lindenmayer import S0LSystem

zoo: List[Tuple[S0LSystem, int]] = [
    # Deterministic L-systems
    (S0LSystem(
        axiom="F-F-F-F",
        productions={
            "F": ["F+FF-FF-F-F+F+F"
                  "F-F-F+F+FF+FF-F"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="-F",
        productions={
            "F": ["F+F-F-F+F"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="F+F+F+F",
        productions={
            "F": ["F+f-FF+F+FF+Ff+FF-f+FF-F-FF-Ff-FFF"],
            "f": ["fffff"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="F-F-F-F",
        productions={
            "F": ["FF-F-F-F-F-F+F"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="F-F-F-F",
        productions={
            "F": ["FF-F-F-F-FF"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="F-F-F-F",
        productions={
            "F": ["FF-F+F-F-FF"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="F-F-F-F",
        productions={
            "F": ["FF-F--F-F"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="F-F-F-F",
        productions={
            "F": ["F-FF--F-F"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="F-F-F-F",
        productions={
            "F": ["F-F+F-F-F"]
        },
        distribution="uniform",
    ), 90),
    # Left- right- rules
    (S0LSystem(
        axiom="F",
        productions={
            "F": ["F+F+",
                  "-F-F"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="F",
        productions={
            "F": ["F+F+F",
                  "F-F-F"]
        },
        distribution="uniform",
    ), 60),
    # Rewriting techniques
    (S0LSystem(
        axiom="F",
        productions={
            "F": ["F+F++F-F--FF-F+",
                  "-F+FF++F+F--F-F"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="-F",
        productions={
            "F": ["FF-F-F+F+F-F-FF""+F+FFF-F+F+FF+F-FF-F-F+F+FF",
                  "+FF-F-F+F+FF+FFF-F-F+FFF-F-FF+F+F-F-F+F+FF"]
        },
        distribution="uniform",
    ), 90),
    # L/R
    # (S0LSystem(
    #     axiom="-L",
    #     productions={
    #         "L": ["LF+RFR+FL-F-LFLFL-FRFR+"],
    #         "R": ["-LFLF+RFRFR+F+RF-LFL-FR"],
    #     },
    #     distribution="uniform",
    # ), 90),
    # (S0LSystem(
    #     axiom="-L",
    #     productions={
    #         "L": ["LFLF+RFR+FLFL-FRF-LFL-FR+F+RF-LFL-FRFRFR+"],
    #         "R": ["-LFLFLF+RFR+FL-F-LF+RFR+FLF+RFRF-LFL-FRFR"],
    #     },
    #     distribution="uniform",
    # ), 90),
    # (S0LSystem(
    #     axiom="L",
    #     productions={
    #         "L": ["LFRFL-F-RFLFR+F+LFRFL"],
    #         "R": ["RFLFR+F+LFRFL-F-RFLFR"]
    #     },
    #     distribution="uniform",
    # ), 90),
    # Branching w/ edge rewriting
    (S0LSystem(
        axiom="F",
        productions={
            "F": ["F[+F]F[-F]F"]
        },
        distribution="uniform",
    ), 25.7),
    (S0LSystem(
        axiom="F",
        productions={
            "F": ["F[+F]F[-F][F]"]
        },
        distribution="uniform",
    ), 20),
    (S0LSystem(
        axiom="F",
        productions={
            "F": ["FF-[-F+F+F]+[+F-F-F]"]
        },
        distribution="uniform",
    ), 22.5),
    # Branching with node rewriting
    (S0LSystem(
        axiom="X",
        productions={
            "X": ["F[+X]F[-X]+X"],
            "F": ["FF"],
        },
        distribution="uniform",
    ), 20),
    (S0LSystem(
        axiom="X",
        productions={
            "X": ["F[+X][-X]FX"],
            "F": ["FF"],
        },
        distribution="uniform",
    ), 25.7),
    (S0LSystem(
        axiom="X",
        productions={
            "X": ["F-[[X]+X]+F[+FX]-X"],
            "F": ["FF"],
        },
        distribution="uniform",
    ), 22.5),
    # Stochastic branching
    (S0LSystem(
        axiom="X",
        productions={
            "X": ["F[+X]FX",
                  "F[-X]FX",
                  "F[+X][-X]FX",
                  "FFX",
                  ],
            "F": ["FF"],
        },
        distribution="uniform",
    ), 20),
    (S0LSystem(
        axiom="F",
        productions={
            "F": ["F[+F]",
                  "F[-F]",
                  "FF"],
        },
        distribution="uniform",
    ), 20),
    (S0LSystem(
        axiom="X",
        productions={
            "X": ["[+FX][-X]FX",
                  "[+X][-FX]FX",
                  "[+FX][-FX]FX"],
            "F": ["FF"],
        },
        distribution="uniform",
    ), 20),
    # moss
    (S0LSystem(
        axiom="F",
        productions={
            "F": ["F[+F]F[-F]F",
                  "F[+F]F",
                  "F[-F]F"],
        },
        distribution="uniform",
    ), 20),
    # (S0LSystem(
    #     axiom="",
    #     productions={
    #         "F": [""]
    #     },
    #     distribution="uniform",
    # ), 90),
]

zoo_systems = [sys for sys, angle in zoo]