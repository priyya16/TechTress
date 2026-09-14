# Problem Statement

## Target audience

- Pharmacovigilance scientists and case-series reviewers who screen spontaneous reports.
- Regulatory-affairs and medical-writing teams assembling ICH M4 Common Technical Document (CTD) dossiers.
- Hackathon judges evaluating a transparent, local prototype rather than a certified product.

## Current problem

Two operational bottlenecks sit on the path from safety data to a filing:

1. **Drug safety signal detection.** Collections such as FDA FAERS contain millions of adverse-event reports. Reviewers must still notice unusual drug-event pairings in that volume. Manual spreadsheet review does not scale, and informal “this looks frequent” judgments are hard to reproduce.
2. **Regulatory submission readiness.** A CTD dossier is organized into five ICH M4 modules covering administrative content, summaries, quality, nonclinical reports, and clinical reports. Missing or incomplete sections delay submission even when the science is otherwise ready.

## Why existing manual approaches are difficult

- Adverse-event files mix missing values, duplicate cases, and inconsistent drug or event names.
- Disproportionality statistics such as Proportional Reporting Ratio (PRR) require a 2x2 table and careful handling of zeros; ad-hoc Excel formulas are easy to get wrong.
- Free-text narratives, when present, are tedious to group by eye.
- Dossier outlines are long checklists maintained in documents that do not automatically score module completeness or prioritize gaps.

## Quantified pain points

The challenge framing emphasizes:

- **Millions** of FAERS-scale reports, which makes unaided signal hunting impractical.
- **Five** ICH M4 CTD modules, each with many expected sections, so a single missed high-impact summary can stall a filing.
- Reviewer time spent on **cleaning, counting, and checklisting** instead of scientific assessment.

## Why this problem matters now

Regulators and industry still rely on spontaneous-report screening as an early warning layer, while CTD remains the shared dossier format for many regions. A lightweight prototype that calculates PRR correctly and surfaces representative CTD gaps can shorten the time to a human review without claiming to replace that review. Patient safety and filing timelines both suffer when signals or missing modules stay hidden in volume.
