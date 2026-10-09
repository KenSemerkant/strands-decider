Feature: Asking from the command line
  `strands-decider ask` renders CLI flags into the same typed request the
  API takes, so a shell script and an HTTP client are the same client.
  Use case: docs/use-cases.md, "decision endpoint for agent stacks".

  Scenario: A noul flag produces a noul answer line
    Given the engine loader returns a stub engine
    When the user runs ask with --noul "is it grounded?"
    Then the exit code is 0
    And the output names the question noul_0

  Scenario: A choice spec picks an option
    Given the engine loader returns a stub engine
    When the user runs ask with --choice "which?=a,b"
    Then the exit code is 0
    And the output names the question choice_0

  Scenario: A malformed choice spec fails with usage help
    Given the engine loader returns a stub engine
    When the user runs ask with --choice "no-separator"
    Then the exit code is 2

  Scenario: Asking nothing is refused
    Given the engine loader returns a stub engine
    When the user runs ask with no question flags
    Then the exit code is 2
