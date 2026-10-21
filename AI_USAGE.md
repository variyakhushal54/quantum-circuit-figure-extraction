# AI usage disclosure

This project was written by Khushalkumar Variya as a course project at OTH
Amberg-Weiden. An AI assistant (Claude) was used later for the tasks listed
below. The author reviewed and decided on each of them. Work the assistant did
not do is listed too.

## Used for

- Changing the arXiv downloader to identify the tool in its User-Agent instead of
  imitating a browser; the contact address is read from `ARXIV_CONTACT_EMAIL`.
- Removing duplicate and unused imports from the pipeline.
- Correcting the page number in the rule-based fallback branch to be 1-based.
- Reading the Tesseract path from an environment variable in the OCR script.
- Writing `scripts/verify_results.py`, which recomputes the headline numbers from
  the committed files.
- Writing `README.md` and `.gitignore`, and adding the results table that lists
  the report's figures next to the recomputed ones.

## Not done by the assistant

- Choosing the papers, the thresholds and the filter design.
- Running the full pipeline. It needs PyMuPDF and access to arXiv, which were not
  available in the environment. Only the adaptive filter was loaded and tested.
- Any change to the report PDF.

## Checks

- The Python files were checked for syntax errors.
- The adaptive filter loaded its learned patterns and scored a test caption.
- `scripts/verify_results.py` was run on the committed files and matches the
  report's totals.
