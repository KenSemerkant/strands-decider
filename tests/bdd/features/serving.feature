Feature: One-shot typed answers over HTTP
  A client posts one state and a set of typed questions to /v1/systemone
  and gets a typed answer per question in a single round trip.
  Use case: docs/use-cases.md, "decision endpoint for agent stacks".

  Scenario: Health reports the serving model
    Given the server is mounted with a stub engine
    When the client checks health
    Then the status is ok and the model is strands-decider-test

  Scenario: A yes/no question comes back as a noul probability
    Given the server is mounted with a stub engine
    When the client asks "is the sky blue?" as a noul question
    Then the answer type is "noul"
    And the answer value is 0.3

  Scenario: A choice question picks the top option with a confidence
    Given the server is mounted with a stub engine
    When the client asks a choice question with options "red, green, blue"
    Then the answer type is "choice"
    And the picked option is the first criterion

  Scenario: A score question returns an expected value with a legend
    Given the server is mounted with a stub engine
    When the client asks a score question with levels "low, mid, high"
    Then the answer type is "score"
    And the answer carries a probability for each level

  Scenario: Mixed question types in one request
    Given the server is mounted with a stub engine
    When the client asks a mixed request of noul, choice and score
    Then each question name has a typed answer
    And usage counts one output token per question

  Scenario: A malformed question body is rejected with 422
    Given the server is mounted with a stub engine
    When the client posts a question of unknown type "wat"
    Then the response status is 422

  Scenario: An empty question set is rejected with 422
    Given the server is mounted with a stub engine
    When the client posts a request with no questions
    Then the response status is 422
