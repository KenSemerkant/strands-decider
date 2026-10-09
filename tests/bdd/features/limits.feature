Feature: Requests beyond the model's budget are refused, not truncated
  The stub engine serves 24 slots and a 512-token window; requests that
  exceed the contract fail loudly with 422 so a client never gets a
  silently-wrong answer.
  Use case: docs/use-cases.md, "decision endpoint for agent stacks".

  Scenario: More options than slots is rejected
    Given the server is mounted with a stub engine
    When the client asks a choice question with 25 options
    Then the response status is 422

  Scenario: A question with empty criteria is rejected
    Given the server is mounted with a stub engine
    When the client posts a choice question with no criteria
    Then the response status is 422

  Scenario: A state that is not a string is rejected
    Given the server is mounted with a stub engine
    When the client posts a request whose state is a number
    Then the response status is 422
