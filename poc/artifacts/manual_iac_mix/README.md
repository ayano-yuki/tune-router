# Manual Technical Log / IaC Collection Kit

Use this folder when you want to collect supplemental technical-log and IaC examples from multiple websites by copy/paste first, then normalize them to the current router labels.

## Why this approach

For mixed source formats (forums, docs, repos, incident reports, blog posts), manual copy/paste is often faster for the first 500-2000 samples than writing one-off scrapers.

## Files in this folder

- `sources_manifest.tsv`: define 9 source slots (name, URL, license, target count).
- `coding_iac_manual_collection.md`: paste raw text blocks from each source.
- `coding_iac_manual_records_template.jsonl`: optional normalized draft lines (one line per record).

## Suggested workflow

1. Fill `sources_manifest.tsv` with your 9 actual sources.
2. Paste raw material into `coding_iac_manual_collection.md` under the right source section.
3. Create record drafts in `coding_iac_manual_records_template.jsonl`.
4. Keep each `text` field under ~4000 chars to match current pipeline constraints.
5. Preserve `source`, `source_url`, and `source_license` for auditability.

## Record quality checklist

- Clearly tied to one of `Storage`, `Network`, `Coding`, `Security`, `Database`, or `General`.
- Not just a repeated generic instruction prompt.
- Has enough detail to distinguish from the `General` label.
- Remove secrets, credentials, private identifiers.

## Notes

- Mixing multiple sources should improve routing robustness and reduce source bias.
