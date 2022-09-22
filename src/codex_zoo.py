from lindenmayer import S0LSystem
import book_zoo

_prompt_zoo = [
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
]

_zoo = [
    (S0LSystem(
     axiom="F-F-F-F",
     productions={
         "F": ["F-F-F-FF"]
     },
     distribution="uniform",
     ), 90),
    (S0LSystem(
        axiom="F-F-F-F",
        productions={
            "F": ["F-F-F+F"]
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
    (S0LSystem(
        axiom="F-F-F-F",
        productions={
            "F": ["F-F-F+F"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="F-F-F-F",
        productions={
            "F": ["F--F+F-F"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="F-F-F-F",
        productions={
            "F": ["F+F-F+F-F"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="F-F-F-F",
        productions={
            "F": ["F+F-F-F+F"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="F-F-F-F",
        productions={
            "F": ["F+F+F-F-F"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="F-F-F-F",
        productions={
            "F": ["F+F-F+F-F"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="F-F-F-F",
        productions={
            "F": ["F+F+F-F+F"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="F-F-F-F",
        productions={
            "F": ["F+F+F+F-F"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="F-F-F-F",
        productions={
            "F": ["F+F-F+F+F"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="F-F-F-F",
        productions={
            "F": ["F-F-F+F+F"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="F-F-F-F",
        productions={
            "F": ["F-F+F+F-F"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="F-F-F-F",
        productions={
            "F": ["F-F+F-F+F"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="F-F-F-F",
        productions={
            "F": ["F-F-F+F-F"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="F-F-F-F",
        productions={
            "F": ["F+F-F-F-F"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="F-F-F-F",
        productions={
            "F": ["F+F-F-F+F"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="F-F-F-F",
        productions={
            "F": ["F+F-F+F-F"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="F-F-F-F",
        productions={
            "F": ["F+F-F+F+F"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="F-F-F-F",
        productions={
            "F": ["F+F+F-F+F"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="F-F-F-F",
        productions={
            "F": ["F+F+F+F-F"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="F-F-F-F",
        productions={
            "F": ["F+F+F-F-F"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="F-F-F-F",
        productions={
            "F": ["F+F+F-F+F"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="F-F-F-F",
        productions={
            "F": ["F+F+F+F+F"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="F-F-F-F",
        productions={
            "F": ["F+F+F-F-F"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="F-F-F-F",
        productions={
            "F": ["F-F+F+F+F"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="F-F-F-F",
        productions={
            "F": ["F-F+F+F-F"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="F-F-F-F",
        productions={
            "F": ["F-F+F-F+F"]
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
    (S0LSystem(
        axiom="F-F-F-F",
        productions={
            "F": ["F-F-F+F+F"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="F-F-F-F",
        productions={
            "F": ["F-F-F+F-F"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="F-F-F-F",
        productions={
            "F": ["F-F-F-F+F"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="F-F-F-F",
        productions={
            "F": ["F-F-F-F-F"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="F-F-F-F",
        productions={
            "F": ["F+F-F-F+F"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="F-F-F-F",
        productions={
            "F": ["F+F-F+F+F"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="F-F-F-F",
        productions={
            "F": ["F+F-F+F-F"
                  "F+F-F-F+F"
                  "F-F+F+F-F"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="F-F-F-F",
        productions={
            "F": ["F+F-F-F+F"
                  "F+F-F+F+F"
                  "F-F+F+F-F"]
        },
        distribution="uniform",
    ), 90),
    (S0LSystem(
        axiom="F-F-F-F",
        productions={
            "F": ["F+F-F+F+F"
                  "F+F-F-F+F"
                  "F-F+F+F-F"]
        },
        distribution="uniform",
    ), 90),
]

specimens = []
angles = []
for specimen, angle in _zoo:
    if specimen not in specimens and \
        all(specimen.productions != x.productions
            for x, _ in book_zoo.zoo):
        specimens.append(specimen)
        angles.append(angle)

print(f"{len(_zoo)} -> {len(specimens)}")
zoo = zip(specimens, angles)
