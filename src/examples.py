_lsystem_book_examples = [
    'F-F-F-F;F~F+FF-FF-F-F+F+FF-F-F+F+FF+FF-F', 90,
    '-F;F~F+F-F-F+F', 90,
    'F+F+F+F;F~F+f-FF+F+FF+Ff+FF-f+FF-F-FF-Ff-FFF,f~fffff',  90,
    'F-F-F-F;F~FF-F-F-F-F-F+F', 90,
    'F-F-F-F;F~FF-F-F-F-FF', 90,
    'F-F-F-F;F~FF-F+F-F-FF', 90,
    'F-F-F-F;F~FF-F--F-F', 90,
    'F-F-F-F;F~F-FF--F-F', 90,
    'F-F-F-F;F~F-F+F-F-F', 90,
    # nondeterminism
    'F;F~F+F+,F~-F-F', 90,
    'F;F~F+F+F,F~F-F-F', 60,
    'F;F~F+F++F-F--FF-F+,F~-F+FF++F+F--F-F', 90,
    '-F;F~FF-F-F+F+F-F-FF+F+FFF-F+F+FF+F-FF-F-F+F+FF,F~+FF-F-F+F+FF+FFF-F-F+FFF-F-FF+F+F-F-F+F+FF', 90, 
    # L/R rules
    '-L;L~LF+RFR+FL-F-LFLFL-FRFR+,R~-LFLF+RFRFR+F+RF-LFL-FR', 90,
    '-L;L~LFLF+RFR+FLFL-FRF-LFL-FR+F+RF-LFL-FRFRFR+,R~-LFLFLF+RFR+FL-F-LF+RFR+FLF+RFRF-LFL-FRFR', 90,
    'L;L~LFRFL-F-RFLFR+F+LFRFL,R~RFLFR+F+LFRFL-F-RFLFR', 90,
    # branching, edge rewriting
    'F;F~F[+F]F[-F]F', 25.7,
    'F;F~F[+F]F[-F][F]', 20,
    'F;F~FF-[-F+F+F]+[+F-F-F]', 22.5,
    # branching, node rewriting
    'F;F~F[+F]F[-F]+F,F~FF', 20,
    'F;F~F[+F][-F]FF,F~FF', 25.7, 
    'F;F~F-[[F]+F]+F[+FF]-F,F~FF', 22.5,
    # stochastic branching
    'F;F~F[+F]FF,F~F[-F]FF,F~F[+F][-F]FF,F~FFF,F~FF', 20,
    'F;F~F[+F],F~F[-F],F~FF', 20,
    'F;F~[+FF][-F]FF,F~[+F][-FF]FF,F~[+FF][-FF]FF,F~FF', 20,
    # moss
    'F;F~F[+F]F[-F]F,F~F[+F]F,F~F[-F]F', 20,
]

lsystem_book_examples = _lsystem_book_examples[::2]  # skip angles
lsystem_book_F_examples = [s.replace("X", "F").replace("L", "F").replace("R", "F")
                           for s in lsystem_book_examples]

_lsystem_chatgpt_examples = [
    # named fractals
    "X;X~F[+X]F[-X]+X,F~F", "Sierpinski Arrowhead Curve",
    "F;F~F[+F]F[-F][F]", "Koch Snowflake",
    "F;F~F[+F]F[-F][F],F~F", "Koch Island",
    "F;F~F[+F]F[-F]F,FF[+F]F", "Hilbert Curve",
    "F;F~F[+F][-F]F[+F], FF[+F]F", "Dragon Curve",
    "F;F~F+F-F-F+F,FF[+F]F", "Pentaplexity",
    "F;F~F[+F]F[-F][F]", "Koch Curve",
    "F;F~FF-[-F+F+F]+[+F-F-F]", "Pentaflake",
    "X;X~YF, Y~X[-FFF][+FFF]FX", "Dragon Plant",
    "X;X~F-[[X]+X]+F[+FX]-X, F~F", "Barnsley Fern",
    "F;F~F[+F]F[-F][F]", "Sierpinski Triangle",
    "F;F~FF-[-F+F+F]+[+F-F-F], F~F[+F]F[-F]F", "Minkowski Island",
    "F;F~FF+[+F-F-F]-[-F+F+F], F~F[+F]F[-F]F", "Sierpinski Arrowhead Curve",
    "F;F~FF+[+F]-[-F], F~F[+F]F[-F]F", "Sierpinski Carpet",
    "F;F~F[+F][-F]F[+F]F, F~F[+F]F", "Gosper Curve",
    "F;F~FF+[+F]-[-F]F, F~F[+F][-F]F", "Hexagonal Gosper Curve",
    "F;F~F[+F][-F][+F][+F], F~F[-F][+F]F", "Pentigree",
    "F;F~F[+F][-F]F[+F][-F]F, F~F[+F]F", "Heighway Dragon",
    "F;F~F[+F][-F]F[+F][-F]F, F~F[+F]F", "Moore Curve",
    "F;F~F[+F]F[-F][F]+F[+F]F[-F]F, X~F+[[X]-X]-F[-FX]+X", "Peano-Gosper Curve",

    # 'new' l-systems
    "X;X~F[+X][-X]FX,F~F", "Zigzag Tree",
    "X;X~F[+X]F[-X]+X,F~F", "Branching Staircase",
    "X;X~F[+X][-X][+X]F,F~F", "Twisted Stalk",
    "X;X~F[+X][-X][FX][+FX][-FX]F,F~F", "Octopus Arms",
    "X;X~F[+X][-X][FX]F[+FX][-FX]F,F~F", "Snowflake",
    "X;X~F[+X]F[-X]+X,F[+F]F[-F]+F", "Leafy Lattice",
    "X;X~F[+X][-X]F[+X]F[-X]X,F[+F]F[-F]", "Triangular Maze",
    "X;X~F[+X][-X]F[+X][FX][-FX]X,F[+F]F[-F]", "Square Maze",
    "X;X~F[+X]F[-X]+F[-FX]+X,F~F", "Diamonds and Triangles",
    "X;X~F[+X][-X]F[+X][-X][+X]X,F~F", "Flower of Life",
]
lsystem_chatgpt_examples = _lsystem_chatgpt_examples[::2]
lsystem_chatgpt_example_names = _lsystem_chatgpt_examples[1::2]
