
from stringsight.core.data_objects import ConversationRecord

# Test case 1: winner in scores dict
c1 = ConversationRecord(
    question_id="c1",
    prompt="p",
    responses=["r1", "r2"],
    model=["model_a", "model_b"],
    scores={"winner": "model_a"},
    meta={}
)
print("Case 1 (winner in scores):", c1.scores)

# Test case 2: winner in meta
c2 = ConversationRecord(
    question_id="c2",
    prompt="p",
    responses=["r1", "r2"],
    model=["model_a", "model_b"],
    scores={},
    meta={"winner": "model_b"}
)
print("Case 2 (winner in meta):", c2.scores)

# Test case 3: tie
c3 = ConversationRecord(
    question_id="c3",
    prompt="p",
    responses=["r1", "r2"],
    model=["model_a", "model_b"],
    scores={"winner": "tie"},
    meta={}
)
print("Case 3 (tie):", c3.scores)

# Test case 4: already list
c4 = ConversationRecord(
    question_id="c4",
    prompt="p",
    responses=["r1", "r2"],
    model=["model_a", "model_b"],
    scores=[{"s": 1}, {"s": 2}],
    meta={"winner": "model_a"}
)
print("Case 4 (already list):", c4.scores)


