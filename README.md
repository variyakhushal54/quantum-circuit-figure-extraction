# Extraction of Quantum Circuit Images from arXiv Papers

A pipeline that downloads arXiv PDFs, finds figure captions that mention quantum
circuits, crops the figure above each caption, and keeps only the images that look
like circuit diagrams. Each kept image is stored with its caption, page and arXiv ID.

Built for the course *Artificial Intelligence for Industrial Applications* at
OTH Amberg-Weiden (Winter 2025). Written by Khushalkumar Variya.

## How it works

1. **Download.** `src/arxiv_downloader.py` fetches each PDF in memory, with a random
   10–20 s delay between requests.
2. **Find captions.** `src/quantum_circuit_pipeline.py` looks for blocks starting with
   "Figure N" or "Fig. N" and keeps captions that contain circuit keywords and no plot keywords.
3. **Crop.** The drawings and images above the caption are merged into one region
   and rendered at 200 DPI. Whitespace is trimmed.
4. **Pre-filter.** For captions with a low keyword score, a code check (OCR), a
   heatmap check (colour saturation) and a plot check (grid, axes, curves) reject obvious non-circuits.
5. **Adaptive judge.** `src/adaptive_filter.py` compares visual features (lines, boxes,
   circles, edge density, colour) with statistics learned from valid examples, using a
   Z-score, and checks caption bigrams against learned phrases.
6. **Structure check.** A rule-based check looks for parallel horizontal wires, gate-sized boxes
   on the wires and vertical CNOT-style connectors. A figure is kept if the adaptive judge
   and the structure check agree, or if the adaptive score is very high.
7. **Save.** Images go to `images_52/`. Metadata goes to `data/circuits_metadata.json`.
   Progress for each paper goes to `paper_list_counts_52.csv`, so a run can be resumed.

The final report is included as `report - documentation.pdf`.

## Results

These are the numbers from the final report (`report - documentation.pdf`). Run
`python scripts/verify_results.py` to recompute the totals from the files in this repository.

| Metric | Final report | Recomputed from repository files |
|---|---|---|
| Papers processed | 1,852 | 1,852 |
| Circuit images saved | 250 | 250 |
| Images per paper | 0.135 | 0.135 |

## Repository layout

```
src/quantum_circuit_pipeline.py   main pipeline (run this)
src/adaptive_filter.py            learned visual and caption scoring
src/arxiv_downloader.py           polite PDF download from arXiv
scripts/learn_from_circuits.py    learns the statistics used by the adaptive filter
scripts/create_dataset_json.py    OCR pass that adds gate labels to the metadata
scripts/generate_csv_report.py    rebuilds the per-paper count file
scripts/evaluate_accuracy.py      precision and recall against a labelled folder
scripts/verify_results.py         recomputes the headline numbers from the included files
data/                             metadata, keyword lists, learned patterns
extra/paper_list_52.txt           arXiv IDs in queue order
paper_list_counts_52.csv          per-paper progress and image counts
```

The extracted figure images are not included. See "Limitations".

## Setup

```bash
pip install -r requirements.txt
```

The pipeline also needs the Tesseract OCR program for the OCR checks. If it is not on your
PATH, set `TESSERACT_CMD` to its location. Without it, the OCR checks are skipped.

Optionally set `ARXIV_CONTACT_EMAIL` to a contact address. It is added to the User-Agent string
that the downloader sends to arXiv.

## Running

```bash
python src/quantum_circuit_pipeline.py            # downloads papers and extracts figures
python scripts/learn_from_circuits.py             # needs data/circuits_final/ (see Limitations)
python scripts/create_dataset_json.py             # optional OCR pass
python scripts/generate_csv_report.py
```

## Limitations

- **Figures come from third-party papers.** Each image belongs to the paper it came from,
  and many papers are not under an open licence. The images are therefore not included here.
  The metadata gives the arXiv ID and page for each figure, so the images can be regenerated.
- **The adaptive judge is trained on its own outputs.** `learn_from_circuits.py` reads the
  images the pipeline kept. Its statistics therefore describe the pipeline's own choices,
  and the score cannot be used as an independent check.
- **The learning script reads `data/circuits_final/`,** a folder with the reviewed images.
  That folder is not in the repository, so the learned patterns cannot be rebuilt from this code alone.
- **Rules are tuned by hand.** Thresholds (keyword lists, line spacing, edge density) were
  adjusted by looking at results. The pipeline has not been evaluated on a held-out set.
- **A fallback path in the pipeline is not used.** When the adaptive filter loads, figures
  are saved by the adaptive branch only. The rule-based branch below it is only a fallback and
  has not been run on its own.
- **Not run end to end here.** The pipeline needs PyMuPDF and access to arXiv. Only the
  adaptive filter was loaded and tested in isolation.
