# dirty import hacks...
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import execution

if __name__ == "__main__":
    execution.unsafe_check("print(3)", timeout=1)