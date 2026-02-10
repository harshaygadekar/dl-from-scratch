---
name: Curriculum Quality Gap
about: Report unclear, inconsistent, or missing learning content in a topic
title: '[QUALITY] Topic XX: Brief gap summary'
labels: curriculum-quality
assignees: ''
---

## Topic Context

**Topic Number**:
<!-- e.g., Topic 24 -->

**Track**:
<!-- Core (01-34) or Bonus (35-38) -->

**Affected Files**:
<!-- e.g., README.md, hints/hint-2-*.md, tests/test_edge.py -->

---

## Quality Gap

**What feels incomplete, unclear, or inconsistent?**
<!-- Be specific about the missing quality bar -->

**Expected quality level**
<!-- What should be present (examples, constraints, success criteria, etc.) -->

**Current behavior/content**
<!-- Describe what exists right now -->

---

## Reproduction Path (Learning Flow)

1.
2.
3.

**Where learners get blocked**
<!-- Exact step where confusion starts -->

---

## Evidence

**Command(s) run:**
```bash
# e.g.
python3 utils/test_runner.py --day 24
python3 scripts/lint_topic_content.py --core-only
```

**Output snippets (if relevant):**
```text
Paste key output lines
```

---

## Suggested Improvement (Optional)

<!-- Add concrete suggestions: missing section, better hint ordering, stronger edge test, etc. -->

---

## Checklist

- [ ] I checked existing hints/introduction files for this topic.
- [ ] I confirmed this is a content quality gap, not only a coding bug.
- [ ] I included enough detail to reproduce the issue.
