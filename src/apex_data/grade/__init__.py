"""Execution graders for the stage-2 pools and the dev evaluation (data plan sections 6 and 11).

swesmith   pristine-image replay of a patch against a calibrated SWE-smith task: `grade_patch(task, patch, run_dir)` ->
           schema.Verdict (see the module docstring); calibration of the gold fix (`calibrate_task`)
reporters  the structured test reporters the replay uses (Go `go test -json`, pytest JUnit XML)
tamper     tamper flags and the test-path split shared by every grader
harbor_tests  the TermGrade terminal verdict (unit U3.2)
"""
