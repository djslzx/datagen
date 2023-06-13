"""
Examples for test domains
"""

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
    'F;F~F+F+F,F~F-F-F', 90,
    'F;F~F+F++F-F--FF-F+,F~-F+FF++F+F--F-F', 90,
    '-F;F~FF-F-F+F+F-F-FF+F+FFF-F+F+FF+F-FF-F-F+F+FF,F~+FF-F-F+F+FF+FFF-F-F+FFF-F-FF+F+F-F-F+F+FF', 90, 
    # L/R rules
    '-L;L~LF+RFR+FL-F-LFLFL-FRFR+,R~-LFLF+RFRFR+F+RF-LFL-FR', 90,
    '-L;L~LFLF+RFR+FLFL-FRF-LFL-FR+F+RF-LFL-FRFRFR+,R~-LFLFLF+RFR+FL-F-LF+RFR+FLF+RFRF-LFL-FRFR', 90,
    'L;L~LFRFL-F-RFLFR+F+LFRFL,R~RFLFR+F+LFRFL-F-RFLFR', 90,
    # branching, edge rewriting
    'F;F~F[+F]F[-F]F', 20,
    'F;F~F[+F]F[-F][F]', 20,
    'F;F~FF-[-F+F+F]+[+F-F-F]', 20,
    # branching, node rewriting
    'F;F~F[+F]F[-F]+F', 20,
    'F;F~F[+F][-F]FF', 20,
    'F;F~F-[[F]+F]+F[+FF]-F', 20,
    'F;F~F[+F]F[-F]+F,F~FF', 20,
    'F;F~F[+F][-F]FF,F~FF', 20,
    'F;F~F-[[F]+F]+F[+FF]-F,F~FF', 20,
    # stochastic branching
    'F;F~F[+F]FF,F~F[-F]FF,F~F[+F][-F]FF,F~FFF,F~FF', 20,
    'F;F~F[+F],F~F[-F],F~FF', 20,
    'F;F~[+FF][-F]FF,F~[+F][-FF]FF,F~[+FF][-FF]FF,F~FF', 20,
    # moss
    'F;F~F[+F]F[-F]F,F~F[+F]F,F~F[-F]F', 20,
]
lsystem_book_examples = [f"{angle};{ex}" for ex, angle in zip(_lsystem_book_examples[::2], _lsystem_book_examples[1::2])]
lsystem_book_det_examples = [s for s in lsystem_book_examples if "," not in s]

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

_regex_handcoded_examples = r"""
phone numbers: \(\d\d\d\) \d\d\d-\d\d\d\d
currencies: $\d?\d?\d(,\d\d\d)+\.\d\d
email: \l+@\l+\.\l\l\l?
zip code: \d\d\d\d\d
ssn: \d\d\d-\d\d-\d\d\d
date in mm/dd/yyyy: 1?\d/\d\d/\d\d\d\d
ip address: (\d{1,3}\.){3}\d{1,3}
http link: https?://\l+(\.\l+)+(/\l)+(\.\l)?
twitter handles: @\l+
time: (0|1)\d:\d\d:\d\d (A|P)M
xml tags: </?\l+>
hashtags: #\w+
us state abbreviations: \p\p
roman numerals: (M|D|C|L|X|V|I)+
binary strings: (0|1)+
hex color codes: #(\d|a|b|c|d|e|f)(\d|a|b|c|d|e|f)(\d|a|b|c|d|e|f)(\d|a|b|c|d|e|f)(\d|a|b|c|d|e|f)(\d|a|b|c|d|e|f)
file paths: (\w+/)+(\w+)?
""".split("\n")
_regex_handcoded_examples = filter(lambda x: x, _regex_handcoded_examples)
_regex_handcoded_examples = list(_regex_handcoded_examples)

regex_handcoded_examples = [line.split(':')[1]
                            for line in _regex_handcoded_examples]

_regex_text_enums = list(filter(lambda x: x, r"""
(Club premises certificate)|(Premises license)|(Temporary event notice)
(YES)|(NO)
(Approved)|(Revoked)|(Refused)|(Withdrawn)|(Surrendered)|(Issued)|(Expired)
(Nonsignificant)|(Significant)
(Positive)|(Negative)
(Chromosome)|(Plasmid)|(Chromosome and Plasmid)|(Not applicable)
(SOCKEYE)|(CHINOOK)
(BERM)|(EURO)|(AMER)
(Yes)|(No)
N|P|A|Z
(small)|(large)
(yes)|(no)|(YES)|(NO)
(Disagree)|(Strongly Disagree)|(Agree)|(Strongly Agree)|(Undecided)
(North)|(East)|(South)|(West)
""".split('\n')))
_regex_text = list(filter(lambda x: x, r"""
\p\p
\p+
\w+@\w+.\w\w\w
\p\l+ \p\. \p\l+
Africa/\p\l+
""".split('\n')))
_regex_text_and_nums = list(filter(lambda x: x, r"""
\d\d_PREM_\d\d\d\d\d
05_PREM_00\d\d\d
\d.\d\d(E-\d\d)|(\d\d\d\d\d)
\d|(NA)
\p\d\d \d\p\p
\p\p\p\p\.\d\d\d\d
HoggPass\d\d
HoggPass\d+
(\d+)|(NA)
Metro\d
\d\d\d \p+ STREET (EAST)|(WEST)
\d\d\d \p+ ROAD
\d\p\p\p\p\d\d\.\d\d\d\d
\d\d\d\d?LDT\d\d
Bin \d
TIER \d+
PAL\d\d\d\d\d\d
\d-th
$\d\d,\d\d\d
$\d\d?\d?,\d\d\d,\d\d\d
$\d,\d\d\d\.\d\d
\d.\d\d\dE-0\d
""".split('\n')))
_regex_nums = list(filter(lambda x: x, r"""
\d\d\d\d\d?
\d\d\d.\d\d\d.\d\d\d.\d\d
\d\d/\d\d/\d\d\d\d
(1|2)\d/0\d/20\d\d
0|1
201\d
20(1|2)\d
2012-06-(0|1|2|3)\d
201\d-(0\d)|1(0|1|2)-((0|1|2)\d)|(3(0|1))
\d\d:\d\d:\d\d
\d
\d\d\d\d
\d\d\d \d\d\d-\d\d\d\d
\d/\d/\d\d\d\d
\d\d.\d\d
\d\d\d\d-\d\d-\d\d
\d+
\d\d\d\d\d\d
-\d
-\d\d.\d\d\d\d
19\d\d
\d\d\.\d
(0|1|2)\d/(0|1)\d/200\d
(1|2|3|4|5)
""".split('\n')))

regex_split = {
    "text enums": _regex_text_enums,
    "text": _regex_text,
    "text and nums": _regex_text_and_nums,
    "nums": _regex_nums
}

if __name__ == "__main__":
    import lindenmayer
    import util

    # cool discovered stuff
    generated = [
        "45;FF;F~FFF+FFF",  # flower
        "60;FF;F~FFF-F",  # abstract
        "60;F;F~F+FFFFF",  # abstract
        "45;F;F~+FFFFFFFF",  # honeycomb
    ]
    lsys = lindenmayer.LSys(step_length=3, render_depth=3, n_rows=128, n_cols=128, kind="deterministic")
    imgs = [lsys.eval(lsys.parse(s)) for s in lsystem_book_det_examples + generated]
    util.plot_image_grid(imgs)