# dirty import hacks...
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import evaluate
import execution


if __name__ == "__main__":
    print(
        evaluate.run_tests(program="print(3)", tests=["def test(): return True"], timeout=10),
        execution.unsafe_check(program="print(3)", timeout=10),
    )