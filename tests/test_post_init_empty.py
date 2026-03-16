
from stringsight.core.data_objects import ConversationRecord

# Test case 1: "empty" list of scores [{}, {}] which from_dataframe produces
# and winner in meta. This should trigger score population.
c1 = ConversationRecord(
    question_id="c1",
    prompt="p",
    responses=["r1", "r2"],
    model=["model_a", "model_b"],
    scores=[{}, {}],  # Effectively empty scores
    meta={"winner": "model_a"}
)
print("Case 1 (effectively empty scores):", c1.scores)

# Test case 2: actually empty scores
c2 = ConversationRecord(
    question_id="c2",
    prompt="p",
    responses=["r1", "r2"],
    model=["model_a", "model_b"],
    scores=[],
    meta={"winner": "model_b"}
)
print("Case 2 (empty list scores):", c2.scores)

# Test case 3: None scores (if that happens)
c3 = ConversationRecord(
    question_id="c3",
    prompt="p",
    responses=["r1", "r2"],
    model=["model_a", "model_b"],
    scores={}, # Treated as empty dict
    meta={"winner": "model_a"}
)
print("Case 3 (empty dict scores):", c3.scores)


