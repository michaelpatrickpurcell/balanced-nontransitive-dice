# Balanced Non-Transitive Dice
Finding balanced sets of non-transitive dice using a SAT solver.

This project relies heavily on the Python module `pysat` and its optional dependencies `pblib` and `aiger`.
The simplest way to get and start using `pysat` is to install the latest stable release of the toolkit from PyPI:
```
$ pip install python-sat[pblib, aiger]
```

## Interesting Four-Dice Sets
The most interesting set of four dice that I've found is:
|       | i   | ii  | iii | iv  |
| :-:   | --: | --: | --: | --: |
| **A** | 0   | 4   | 7   | 7   |
| **B** | 3   | 3   | 6   | 6   |
| **C** | 2   | 2   | 5   | 9   |
| **D** | 1   | 1   | 8   | 8   |

Notice that:
  - P{A > B} = P{B > C} = P{C > D} = P{D > A} = 5/8
  - P{A > C} = P{B > D} = 1/2

So, we have a nontransitive cycle of length four and the dice that are not adjacent in that cycle are evenly matched.

The following table describes the probability that the die in each row beats the die in each column.

|       | A   | B   | C   | D   |
| :-:   | :-: | :-: | :-: | :-: |
| **A** |  x  | 5/8 | 4/8 | 3/8 |
| **B** | 3/8 |  x  | 5/8 | 4/8 |
| **C** | 4/8 | 3/8 |  x  | 5/8 |
| **D** | 5/8 | 4/8 | 3/8 |  x  |

Contrast this set with [Efron's Dice](https://en.wikipedia.org/wiki/Intransitive_dice#Efron's_dice). In that set, while the bias on the balanced nontransitive cycle is stronger, only one of the two nonadjacent pairs are evenly matched.
## Interesting Five-Dice Sets
The most interesting set of five dice that I've found is:
|       | i   | ii  | iii | iv  | v   | vi  |
| :-:   | --: | --: | --: | --: | --: | --: |
| **A** | 0   | 0   | 9   | 9   | 9   | 9   |
| **B** | 2   | 6   | 8   | 8   | 8   | 8   |
| **C** | 5   | 5   | 5   | 5   | 7   | 11  |
| **D** | 4   | 4   | 4   | 4   | 13  | 13  |
| **E** | 1   | 3   | 3   | 10  | 10  | 12  |

Notice that:
  - P{A > B} = P{B > C} = P{C > D} = P{D > E} = P{E > A} = 2/3
  - P{A > C} = P{C > E} = P{E > B} = P{B > D} = P{D > A} = 5/9

So, we have two balanced cycles with different biases. Furthermore the biases are related in such a way that each die is:
  - very strong against one die with P{X > X+1} = 6/9.
  - moderately strong against one die with P{X > X+2} = 5/9.
  - moderately weak against one die with P{X > X+3} = 4/9.
  - very weak against one die with P{X > X+4} = 3/9.

The following table describes the probability that the die in each row beats the die in each column.

|       | A   | B   | C   | D   | E   |
| :-:   | :-: | :-: | :-: | :-: | :-: |
| **A** |  x  | 6/9 | 5/9 | 4/9 | 3/9 |
| **B** | 3/9 |  x  | 6/9 | 5/9 | 4/9 |
| **C** | 4/9 | 3/9 |  x  | 6/9 | 5/9 |
| **D** | 5/9 | 4/9 | 3/9 |  x  | 6/9 |
| **E** | 6/9 | 5/9 | 4/9 | 3/9 |  x  |
