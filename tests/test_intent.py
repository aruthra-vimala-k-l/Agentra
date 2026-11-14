from app.intent import extract_intent, missing_slots, next_clarifying_question


def test_extracts_occasion_category_and_budget_range():
    slots = extract_intent(
        "It's my boyfriend's one-year anniversary. My budget is ₹6,000-7,000. "
        "I want something in gold that feels permanent."
    )
    assert slots.occasion == "anniversary"
    assert slots.category == "jewellery"
    assert slots.budget_max == 7000.0
    assert slots.wants_durability is True


def test_extracts_diy_goal():
    slots = extract_intent("I have a school project due tomorrow, need to build a model rocket, budget 500")
    assert slots.category == "diy"
    assert slots.occasion == "school project"
    assert slots.budget_max == 500.0


def test_missing_slots_and_clarifying_question_order():
    slots = extract_intent("I need a gift")
    missing = missing_slots(slots)
    assert "category" in missing
    assert "budget_max" in missing
    q = next_clarifying_question(slots, questions_asked=0)
    assert q is not None


def test_clarifying_question_budget_capped():
    slots = extract_intent("hello")
    assert next_clarifying_question(slots, questions_asked=2) is None


def test_existing_slots_are_not_overwritten_by_a_later_turn():
    slots = extract_intent("budget 6000, jewellery")
    updated = extract_intent("actually electronics", existing=slots)
    # category was already resolved to jewellery -- a later, unrelated
    # mention of "electronics" must not silently overwrite it.
    assert updated.category == "jewellery"


def test_bare_number_alone_with_no_category_and_no_hint_is_not_a_budget():
    """A bare number with nothing else to anchor it to (no category resolved,
    no `expecting` hint) stays ambiguous -- it's not treated as a budget."""
    slots = extract_intent("3000")
    assert slots.budget_max is None


def test_bare_number_is_accepted_as_budget_when_expected():
    """When the orchestrator knows the last question was the budget
    question, a plain '3000' with no keyword must resolve to a ₹3000 budget
    instead of being silently dropped (this was the original bug: typing a
    bare number in reply to 'What's your budget?' did nothing)."""
    base = extract_intent("electronics")
    slots = extract_intent("3000", existing=base, expecting="budget_max")
    assert slots.budget_max == 3000.0


def test_bare_number_alongside_a_category_in_one_line_is_accepted_as_budget():
    """The single-line phrasing 'electronics, 5000' or 'electronics 5000' is
    at least as common as the two-turn version -- a bare number sitting next
    to an already-resolved category is unambiguous enough to count as the
    budget without needing a '₹'/'budget' keyword too."""
    slots = extract_intent("electronics, 5000")
    assert slots.category == "electronics"
    assert slots.budget_max == 5000.0

    slots2 = extract_intent("electronics 5000")
    assert slots2.budget_max == 5000.0
