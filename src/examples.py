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

# note: most of these names are wrong and have nothing to do with the given l-system
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

_regex_chatgpt_examples = r"""
    Matching all email addresses:
    [a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}

    Matching all phone numbers:
    (\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{10})

    Matching all URLs:
    ^(https?|ftp)://(-\.)?([^\s/?\.#-]+\.?)+(/[^\s]*)?$

    Matching all social security numbers:
    ^\d{3}-\d{2}-\d{4}$

    Matching all IP addresses:
    \b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b

    Matching all dates in the format MM/DD/YYYY:
    ^(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/([0-9]{4})$

    Matching all HTML tags:
    <([a-z]+)([^<]+)*(?:>(.*)<\/\1>|\s+\/>)

    Matching all hexadecimal color codes:
    ^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$

    Matching all words that start and end with the same letter:
    \b(\w)\w*\1\b

    Matching all sentences that contain a specific word:
    ^(?=.*\bword\b).*$
    
    Matching all strings that start with "hello" followed by any number of characters:
    ^hello.*$

    Matching all strings that contain only letters (upper or lowercase):
    ^[a-zA-Z]+$

    Matching all strings that contain at least one uppercase letter and one digit:
    ^(?=.*[A-Z])(?=.*\d).*$

    Matching all strings that contain at least one word starting with "a" and ending with "z":
    \b\w*a\w*z\w*\b

    Matching all strings that are palindromes (reads the same backwards as forwards):
    ^(.)(.?)(.?)(.?)(.?)(.?).?\6\5\4\3\2\1$

    Matching all strings that are valid variable names (start with a letter or underscore, followed by any number of letters, digits, or underscores):
    ^[a-zA-Z_][a-zA-Z0-9_]*$

    Matching all strings that contain a repeated sequence of three or more characters:
    (\w)\1{2,}

    Matching all strings that are exactly 10 characters long and contain only letters and digits:
    ^[a-zA-Z0-9]{10}$

    Matching all strings that contain a sequence of four or more consecutive digits:
    \d{4,}

    Matching all strings that start and end with the same two characters:
    ^(..).*\1$
"""
