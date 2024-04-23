from trace_verify_state import TraceAbstract, verify_traces


if __name__ == "__main__":
    example_line_sat = "5 -1 -3 0 -3 4 5 0 3 1 -5 0 -4 6 -3 0 -6 -2 1 0 2 -6 3 0 -4 -2 3 0 -4 -2 -1 0 -6 4 1 0 6 -4 -3 0 4 1 -2 0 2 -4 3 0 -2 -1 -6 0 4 6 3 0 5 6 4 0 3 -6 4 0 -1 -6 -2 0 5 -6 -2 0 -6 -3 -2 0 -1 -2 -4 0 4 3 -6 0 5 6 -3 0 3 1 4 0 4 -6 -5 0 [SEP] | Decide 4 | D 4 | Decide 2 | D 4 D 2 | UP 3 | D 4 D 2 3 | UP 6 | D 4 D 2 3 6 | BackTrack -2 | D 4 -2 | UP 3 | D 4 -2 3 | UP 6 | D 4 -2 3 6 | Decide 5 | D 4 -2 3 6 D 5 | SAT"
    example_line_unsat = "-7 4 -5 0 6 -4 -1 0 3 5 -2 0 -1 -5 2 0 3 4 -5 0 -7 6 -4 0 -2 7 1 0 5 -7 -2 0 6 3 2 0 6 4 1 0 -6 -4 2 0 -4 2 6 0 -4 -7 -2 0 -7 -3 4 0 -3 6 4 0 4 -3 -6 0 -4 1 2 0 -7 -5 6 0 -4 -2 7 0 3 2 -4 0 3 -4 -6 0 3 1 -7 0 -6 1 4 0 2 -3 6 0 -6 5 4 0 -2 4 1 0 -2 3 1 0 -4 1 3 0 -3 6 7 0 [SEP] | Decide 4 | D 4 | Decide 2 | D 4 D 2 | UP -7 | D 4 D 2 -7 | BackTrack -2 | D 4 -2 | UP -6 | D 4 -2 -6 | BackTrack -4 | -4 | Decide 3 | -4 D 3 | UP -7 | -4 D 3 -7 | UP 6 | -4 D 3 -7 6 | BackTrack -3 | -4 -3 | UP -5 | -4 -3 -5 | UP -2 | -4 -3 -5 -2 | UP 6 | -4 -3 -5 -2 6 | BackTrack 0 | UNSAT"
    example_incorrect_trace = "5 -1 -3 0 -3 4 5 0 3 1 -5 0 -4 6 -3 0 -6 -2 1 0 2 -6 3 0 -4 -2 3 0 -4 -2 -1 0 -6 4 1 0 6 -4 -3 0 4 1 -2 0 2 -4 3 0 -2 -1 -6 0 4 6 3 0 5 6 4 0 3 -6 4 0 -1 -6 -2 0 5 -6 -2 0 -6 -3 -2 0 -1 -2 -4 0 4 3 -6 0 5 6 -3 0 3 1 4 0 4 -6 -5 0 [SEP] | Decide 4 | D 4 | Decide 2 | D 4 D 2 | UP 3 | D 4 D 2 3 | UP 6 | D 4 D 2 3 1 | BackTrack -2 | D 4 -2 | UP 3 | D 4 -2 3 | UP 6 | D 4 -2 3 6 | Decide 5 | D 4 -2 3 6 D 5 | SAT"
    example_incorrect_trace_2 = "5 -1 -3 0 -3 4 5 0 3 1 -5 0 -4 6 -3 0 -6 -2 1 0 2 -6 3 0 -4 -2 3 0 -4 -2 -1 0 -6 4 1 0 6 -4 -3 0 4 1 -2 0 2 -4 3 0 -2 -1 -6 0 4 6 3 0 5 6 4 0 3 -6 4 0 -1 -6 -2 0 5 -6 -2 0 -6 -3 -2 0 -1 -2 -4 0 4 3 -6 0 5 6 -3 0 3 1 4 0 4 -6 -5 0 [SEP] | Decide 4 | D 4 | Decide 2 | D 4 D 2 | UP 3 | D 4 D 2 3 | UP 6 | D 4 D 2 3 1 | Now is the winter of our discontent made glorious summer by this son of York. | D 4 -2 3 6 | Decide 5 | D 4 -2 3 6 D 5 | Røten nik Akten Di | D 4 -2 3 6 D 5 | Wi nøt trei a høliday in Sweden this yër ? | MY CURRENT STATE IS THAT I AM BOTHERED | See the løveli lakes | Gruntled. | Far out in the uncharted backwaters of the unfashionable end of the western spiral arm of the galaxy lies a small, unregarded, yellow sun.  Orbiting this at a distance of roughly 93 million miles is an insignificant little blue-green planet whose ape-decended inhabitants are so amazingly primitive that they still think ChatGPT is a pretty neat idea. | Dead. | SAT"

    trace = TraceAbstract(example_line_sat)
    trace_incorrect = TraceAbstract(example_incorrect_trace)

    print(f"Raw formula: {trace.raw_formula}")
    print(f"Formula: {trace.formula}")
    print(f"States: {trace.states}")
    print(f"Actions: {trace.actions}")
    print(f"Solution: {trace.solution}")

    # test state transitions;
    test_state = "D 4 D 3 2 -5"

    print(f"\nTest state: {test_state}")

    t_decide = trace.Decide(test_state, "6")
    print(f"Decide 6: {t_decide}")
    assert t_decide == "D 4 D 3 2 -5 D 6", "Decide failed"

    t_up = trace.UP(test_state, "-1")
    print(f"UP 1: {t_up}")
    assert t_up == "D 4 D 3 2 -5 -1", "UP failed"

    t_backtrack = trace.BackTrack(test_state, "-3")
    print(f"Backtrack -3: {t_backtrack}")
    assert t_backtrack == "D 4 -3", "Backtrack failed"

    # same test, but now test eval from parsing action
    print("\nTesting eval from parsing action:")

    action, arg = "Decide 6".split(" ")
    t_decide = eval(f"trace.{action}(test_state, arg)")
    print(f"Decide 6: {t_decide}")
    assert t_decide == "D 4 D 3 2 -5 D 6", "Decide failed"

    action, arg = "UP -1".split(" ")
    t_up = eval(f"trace.{action}(test_state, arg)")
    print(f"UP 1: {t_up}")
    assert t_up == "D 4 D 3 2 -5 -1", "UP failed"

    action, arg = "BackTrack -3".split(" ")
    t_backtrack = eval(f"trace.{action}(test_state, arg)")
    print(f"Backtrack -3: {t_backtrack}")
    assert t_backtrack == "D 4 -3", "Backtrack failed"

    print("\n\n")

    print(f"Assignment: {trace.get_assignment()}")
    print(f"Valid assignment: {trace.is_valid_assignment()}")
    print(f"Formula satisfied: {trace.is_formula_satisfied()}")
    print(f"Solve SAT / correct answer: {trace.solve_sat()}")

    print("\n\n")

    print(f"Parsing action 'Decide 4' as: {trace.parse_action('Decide 4')}")

    print(f"Is korrekt trace korrekt: {trace.oll_korrekt()}")
    print(f"Is unkorrekt trace korrekt: {trace_incorrect.oll_korrekt()}")

    # test a bunch
    correct_pred, correct_sat, correct_unsat, all_correct, total_sat, total_unsat, total = verify_traces(
        [
            example_line_sat,
            example_line_unsat,
            example_incorrect_trace,
            example_incorrect_trace_2,
        ]
    )

    print(
        f"Total Fully Correct: {all_correct}/{total} ({all_correct / total * 100:.2f}%)"
    )
    print(
        f"SAT/UNSAT Correct: {correct_pred}/{total} ({correct_pred / total * 100:.2f}%)"
    )
    print(
        f"Correct SAT: {correct_sat}/{total_sat} ({correct_sat / total_sat * 100:.2f}%)"
    )
    print(
        f"Correct UNSAT: {correct_unsat}/{total_unsat} ({correct_unsat / total_unsat * 100:.2f}%)"
    )
