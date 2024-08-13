#!/bin/zsh

# Problem 3.1
run_problem_3_1() {
    echo "Running Problem 3.1..."
    python hw1.py --game rock_paper_superscissors.json --problem 3.1
    python hw1.py --game kuhn_poker.json --problem 3.1
    python hw1.py --game leduc_poker.json --problem 3.1
}

# Problem 3.2
run_problem_3_2() {
    echo "Running Problem 3.2..."
    python hw1.py --game rock_paper_superscissors.json --problem 3.2
    python hw1.py --game kuhn_poker.json --problem 3.2
    python hw1.py --game leduc_poker.json --problem 3.2
}

# Problem 3.3
run_problem_3_3() {
    echo "Running Problem 3.3..."
    python hw1.py --game rock_paper_superscissors.json --problem 3.3
    python hw1.py --game kuhn_poker.json --problem 3.3
    python hw1.py --game leduc_poker.json --problem 3.3
}

while getopts "123" opt; do
    case $opt in
        1)
            run_problem_3_1
            ;;
        2)
            run_problem_3_2
            ;;
        3)
            run_problem_3_3
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            echo "Usage: ./compile_hw1.sh -1 (for 3.1) -2 (for 3.2) -3 (for 3.3)"
            ;;
    esac
done
