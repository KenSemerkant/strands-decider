Feature: Gating a tool call on two typed judgements
  An agent asks the decider two yes/no questions before letting a tool
  call run: are the arguments grounded, and is it premature? Typed
  probabilities become ordinary control flow.
  Use case: docs/use-cases.md, "tool-call gating for agents".

  Scenario: Grounded and not premature — the tool call runs
    Given a gate with a decider answering grounded 1.0 and premature 0.0
    When the agent proposes a tool call
    Then the tool call is returned unchanged

  Scenario: Ungrounded arguments — the agent is guided to ask instead
    Given a gate with a decider answering 0.0
    When the agent proposes a tool call
    Then the action is a Guide
    And the feedback tells the agent to confirm values with the user

  Scenario: Premature call — the agent is told to clarify first
    Given a gate with a decider answering grounded 1.0 and premature 0.9
    When the agent proposes a tool call
    Then the action is a Guide
    And the feedback tells the agent to clarify with the user first

  Scenario: The state names the tool and its arguments
    Given a gate with a decider answering 1.0
    When the agent proposes a tool call
    Then the rendered state contains the tool name and its arguments
