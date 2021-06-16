# Balanced Non-Transitive Dice
Finding balanced sets of non-transitive dice using a SAT solver.

## Interesting Five-Dice Sets
The most interesting set of five dice that I've found is:
| Die | i | ii | iii | iv | v | vi |
| :-: | --: | --: | --: | --: | --: | --: |
| A   | 1   | 3   | 3   | 10  | 10  | 12  |
| B   | 0   | 0   | 9   | 9   |  9  | 9   |
| C   | 2   | 6   | 8   | 8   |  8  | 8   |
| D   | 5   | 5   | 5   | 5   |  7  | 11  |
| E   | 4   | 4   | 4   | 4   |  13 | 13  |

Notice that:
  - P(A > B) = P(B > C) = P(C > D) = P(D > E) = P(E > A) = 2/3
  - P(A > C) = P(C > E) = P(E > B) = P(B > D) = P(D > A) = 5/9

So, we have two balanced cycles with different biases.
