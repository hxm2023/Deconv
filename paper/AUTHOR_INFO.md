# Author and submission metadata (fill before submission)

The manuscript is scientifically complete, but Ultramicroscopy requires the
following information. Replace every bracket; do not submit with placeholders.

## Authors

| # | Full name | Affiliation | Email | ORCID |
|---|---|---|---|---|
| 1 | [Name] | [Department, Institution, City, Country] | [email] | [ORCID] |
| 2 | [Name] | [Affiliation] | [email] | [ORCID] |

Corresponding author: [Name], [email], [postal address]

## Manuscript metadata

- Title: Dose-agnostic neural deconvolution for low-dose fluorescence microscopy
- Article type: Research paper
- Keywords: fluorescence microscopy; deconvolution; low-dose imaging; deep learning; dose-agnostic restoration; uncertainty
- Suggested reviewers (optional): [3-5 names, affiliations, emails; no conflicts]
- Opposed reviewers (optional): [names with reason]

## Required statements

- Funding: [grant numbers and agencies, or "This research received no external funding."]
- Conflict of interest: [declare or state "The authors declare no competing interests."]
- Data availability: public datasets BBBC014/015/016/018/039 and AI4Life-MDC25
  (Zenodo 15624741); code and JSON results will be released at [repository URL].
- Code availability: [repository URL/DOI]
- CRediT author contributions: [conceptualization; methodology; software;
  validation; formal analysis; investigation; data curation; writing—original
  draft; writing—review and editing; visualization; supervision; funding
  acquisition]
- Acknowledgements: [optional]
- Use of AI tools: [declare if required by the journal]

## Files to upload

- `paper/main.pdf` (12 pages; readable 10 pt version)
- `paper/si.pdf` (9 pages)
- `paper/main_elsevier.pdf` (22-page Elsevier preprint with line numbers and
  `elsarticle-num` references; use this for Editorial Manager)
- Cover letter (`paper/cover_letter.md` → PDF)
- Highlights (5 bullets, `paper/highlights.md`)
- Graphical abstract (optional; can reuse Fig. 1 or Fig. 4)
- Source files: `main.tex`, `si.tex`, `references.bib`, `figures/*.pdf`,
  `tables/*.tex`, `numbers.tex`, `real_numbers.tex`, `fmd_numbers.tex`
- Data/code archive: repository or Zenodo DOI

## Format note

`main.tex` is the content source. `src/make_elsevier.py` regenerates
`main_elsevier.tex` with the Ultramicroscopy frontmatter, keywords, line
numbers, and `elsarticle-num` references. Fill the `[Author ...]`,
`[Department, Institution]`, and declaration placeholders in
`main_elsevier.tex` from the metadata above, then rebuild before submission.
