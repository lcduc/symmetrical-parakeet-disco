# Project 3 — Agent Build Prompts
## `hr-cv-matcher`: 18 copy-paste steps

Same format as the Project 1 & 2 prompt files: each step is self-contained — an agent working from only that block (with read access to the repo it's building) should be able to execute it and know when it's done, without needing the full implementation plan as context. Steps are in build order and match the week-by-week plan in `project3_implementation_plan.md`.

Paste one step at a time. Don't paste the next step until the current one's acceptance criteria are met.

---

### Step 1 — Repo scaffold + Docker Compose

```
You are building `hr-cv-matcher`, an HR resume-parsing and JD-matching tool.
This is Step 1 of 18. Do not build anything beyond what's listed here.

Create this exact directory structure (empty __init__.py files where shown,
no logic yet):

hr-cv-matcher/
├── docker-compose.yml
├── .env.example
├── requirements.txt
├── README.md
├── __init__.py
├── main.py
├── config.py
├── db.py
├── models/__init__.py
├── services/__init__.py
├── prompts/__init__.py
├── utils/__init__.py
└── api/__init__.py
├── app/
├── eval/
├── tests/
└── .github/workflows/

docker-compose.yml needs two services only: `postgres` (image postgres:16,
port 5434:5432, env POSTGRES_DB/USER/PASSWORD) and `api` (build: ., runs
uvicorn app.main:app on port 8002:8000, depends_on postgres). Do NOT add
Qdrant, Redis, or any other service — this project doesn't need them (see
README note you'll add in Step 18 for why).

.env.example needs: POSTGRES_DSN, LLM_PROVIDER, LLM_API_KEY,
EMBEDDING_MODEL_NAME, VLM_DEVICE.

requirements.txt: fastapi, uvicorn, pydantic, pydantic-settings, sqlalchemy,
asyncpg, alembic, anthropic, sentence-transformers, pytesseract, opencv-python,
pymupdf, docxtpl, transformers, torch, streamlit, pytest, pytest-asyncio, ruff.

DO NOT write main.py's actual FastAPI app yet, and DO NOT write any
service or model logic — that starts in Step 2.

Acceptance criteria:
- `docker compose up postgres` starts a healthy Postgres container on port 5434
- Directory structure matches exactly what's listed above
- requirements.txt installs cleanly in a fresh venv
```

---

### Step 2 — Database schema + migrations

```
Step 2 of 18. The repo scaffold from Step 1 exists. Do not touch
services/, api/, or prompts/ in this step.

Write models/db.py with SQLAlchemy 2.0-style ORM models for five
tables: candidates, jobs, matches, interview_questions, embedding_cache.

Exact columns:
- candidates: id (UUID pk), full_name (str, default ""), email (str,
  default ""), phone (str|None), raw_file_path (str), structured_json
  (JSONB, default {}), extraction_status (str, default "pending"),
  extraction_attempts (int, default 0), created_at (datetime)
- jobs: id (UUID pk), title (str), jd_text (str), requirements_json (JSONB),
  created_at (datetime)
- matches: id (UUID pk), candidate_id (FK), job_id (FK), passed_hard_filters
  (bool), hard_filter_notes (str), soft_fit_score (float), overall_score
  (float), explanation (str), created_at (datetime) — UNIQUE(candidate_id, job_id)
- interview_questions: id (UUID pk), candidate_id (FK), job_id (FK),
  questions_json (JSONB), created_at (datetime)
- embedding_cache: cache_key (str pk), vector (ARRAY(Float)), created_at (datetime)

Set up Alembic (alembic.ini + migrations/ directory) and generate the
initial migration from these models. Write db.py with an async
session factory (asyncpg + SQLAlchemy async engine) reading POSTGRES_DSN
from config.py — also write config.py now with a pydantic-settings
Settings class covering every var in .env.example plus
max_extraction_retries (default 3) and hard_filter_soft_cap (default 0.4).

DO NOT write Pydantic domain models (CandidateProfile etc.) yet — that's
Step 3. These are ORM models only.

Acceptance criteria:
- `alembic upgrade head` against the Step 1 Postgres container creates all
  5 tables with correct columns and the UNIQUE constraint on matches
- db.py's session factory can open and close a session without error
```

---

### Step 3 — Pydantic domain models + PII-exclusion tests

```
Step 3 of 18. Database layer from Step 2 exists. Do not write any service
logic yet.

Create models/candidate.py with:
- WorkExperience(company, title, start_date, end_date: str|None,
  description="")
- EducationEntry(institution, degree, field_of_study: str|None, level,
  graduation_year: int|None)
- CandidateProfile(candidate_id="", full_name, email, phone: str|None,
  skills: list[str], work_history: list[WorkExperience],
  education: list[EducationEntry], certifications: list[str],
  years_experience_total: float=0.0) with two methods,
  experience_summary() and education_summary(), that concatenate the
  relevant fields into plain text strings for embedding later.

IMPORTANT: CandidateProfile must NOT have fields for date_of_birth, age,
photo, nationality, marital_status, or gender. This is intentional and
permanent — do not add them even if it seems more "complete." Add a
docstring on the class stating why.

Create models/job.py with JobRequirements(job_id="", title, jd_text,
must_have_skills: list[str], nice_to_have_skills: list[str],
min_years_experience: float=0.0, required_education_level: str|None,
required_certifications: list[str], responsibilities_text="").

Create models/match.py with MatchResult(candidate_id, job_id,
passed_hard_filters: bool, hard_filter_notes: str, soft_fit_score: float,
overall_score: float, explanation: str).

Write tests/test_candidate_schema.py with a test that asserts
CandidateProfile.model_fields does NOT contain any of: date_of_birth, age,
photo, nationality, marital_status, gender. This test should fail loudly
if anyone adds one of these fields later.

Acceptance criteria:
- All three model files import cleanly with no circular imports
- The PII-exclusion test passes
- experience_summary() and education_summary() return non-empty strings
  given a populated CandidateProfile
```

---

### Step 4 — OCR router + VLM client (ported from Project 2)

```
Step 4 of 18. Domain models from Step 3 exist.

Create services/ocr_router.py with:
- an OCRResult dataclass (text: str, mean_confidence: float,
  layout_complexity: float)
- run_fast_ocr(file_path) using pytesseract on a preprocessed
  (grayscale, denoised, Otsu-thresholded via OpenCV) image, returning
  OCRResult with mean_confidence computed from pytesseract's per-word
  confidence data
- _estimate_layout_complexity(ocr_data) — a heuristic based on the spread
  of distinct x-positions in the OCR word boxes (wide spread = more
  complex/multi-column layout), normalized to 0-1
- quality_score(result) = mean_confidence * (1 - 0.5 * layout_complexity)
- run_vlm_ocr(file_path) that calls a VLMClient (Step 4b below)

Create services/vlm_client.py with a VLMClient class wrapping
Qwen/Qwen2-VL-2B-Instruct via transformers, 4-bit quantized, with an
extract_text(file_path) method that runs the model with a transcription
prompt and returns decoded text.

Create services/ingestion_service.py with an IngestionService class:
extract_text(file_path) — if the file is a PDF, try
_try_native_text_layer() first (PyMuPDF, require >=200 chars of text or
return None), otherwise/fallback to _run_ocr_pipeline() which calls
run_fast_ocr(), checks quality_score() against a 0.85 threshold, and
falls back to run_vlm_ocr() if below threshold.

DO NOT write the extraction service (LLM-based structured extraction) yet
— that's Step 5. This step is purely "get raw text out of a file."

Acceptance criteria:
- Given a digital PDF with a text layer, IngestionService.extract_text
  returns the native text without invoking OCR
- Given a scanned image file, it runs through run_fast_ocr and, if
  quality_score is below 0.85, falls through to run_vlm_ocr
- No function referenced in this step is left unimplemented or stubbed
```

---

### Step 5 — Structured extraction with self-heal retry

```
Step 5 of 18. Ingestion pipeline from Step 4 exists.

Create prompts/extraction_prompt.py with:
- SCHEMA_JSON = CandidateProfile.model_json_schema()
- build_extraction_prompt(resume_text) — instructs the LLM to extract
  only schema fields, explicitly forbidding date of birth, age, photo,
  nationality, marital status, gender even if present in source text,
  and to respond with JSON only
- build_correction_prompt(resume_text, validation_error) — same as above
  plus a list of the specific field errors from the ValidationError,
  asking for corrected JSON

Create services/extraction_service.py with ExtractionService(llm):
extract(resume_text) -> tuple[CandidateProfile|None, int]. Loop up to
MAX_RETRIES=3 attempts: call llm.complete_structured(prompt), try
CandidateProfile.model_validate_json(raw); on ValidationError, build a
correction prompt using the specific error and retry. Return (None, 3)
if all retries fail — never raise out of this method. Log at INFO on
each attempt and at ERROR when retries are exhausted.

Create utils/pii_scrub.py with scrub_free_text(text) — regex-strips
any of "date of birth", "age: <number>", "marital status", "nationality"
patterns from free text, as defense-in-depth against these leaking into
a description field even though the schema excludes them structurally.

Write tests/test_extraction_service.py using a mocked LLMAdapter
(AsyncMock) covering: (1) succeeds on first attempt, (2) fails once then
succeeds on retry with a corrected prompt, (3) exhausts all 3 retries and
returns (None, 3). Write tests/test_pii_scrub.py covering that
scrub_free_text removes a marital-status mention while preserving
surrounding text.

DO NOT wire this into an API endpoint yet — that's Step 6.

Acceptance criteria:
- All extraction_service tests pass using mocked LLM calls, no live API
  calls in tests
- pii_scrub test passes
- extract() never raises; the None-retries-exhausted path is exercised
  by a test
```

---

### Step 6 — Upload API endpoint (end-to-end wiring)

```
Step 6 of 18. Extraction service from Step 5 exists.

Create services/llm_adapter.py with an abstract LLMAdapter (async
complete(prompt, max_tokens, temperature) and async
complete_structured(prompt)) and one concrete AnthropicAdapter
implementation using the anthropic SDK's AsyncAnthropic client,
model="claude-haiku-4-5-20251001" as the default.

Create api/candidates.py with:
- POST /candidates/upload (202 status): saves the uploaded file to
  data/uploads/, inserts a `pending` Candidate row, kicks off
  _process_candidate as a FastAPI BackgroundTask, returns
  {candidate_id, status: "pending"} immediately. Do NOT use a job queue
  (Celery, arq, procrastinate) — a BackgroundTask is the deliberate
  choice at this scale.
- _process_candidate(candidate_id, file_path): runs IngestionService then
  ExtractionService, applies scrub_free_text to each work_history
  description, updates the Candidate row to extraction_status="ok" with
  structured_json populated, or "needs_review" if extraction returned None.
- GET /candidates/{candidate_id}: returns {candidate_id, status, profile}
  or 404 if not found.

Wire this router into main.py.

Acceptance criteria:
- POSTing a real resume PDF to /candidates/upload returns 202 immediately
- Polling GET /candidates/{id} shows status transition from "pending" to
  "ok" (or "needs_review") once the background task completes
- Manually test with 10-15 sample resumes (mix of digital PDFs and
  scanned images) and confirm most reach "ok" status
```

---

### Step 7 — Template service + company template

```
Step 7 of 18. Working extraction pipeline from Step 6.

Build templates/company_resume_template.docx by hand in Word (or
LibreOffice) using docxtpl's Jinja syntax: {{ full_name }}, {{ email }},
{{ phone }}, a {% for skill in skills %}{{ skill }}{% endfor %} list, a
{% for job in work_history %}{{ job.title }} at {{ job.company }}
({{ job.start_date }} - {{ job.end_date }}): {{ job.description }}
{% endfor %} section, and equivalent for education and certifications.
This is a one-time manual asset — note in a code comment that it is not
meant to be generated programmatically.

Create services/template_service.py with TemplateService
(template_path defaulting to templates/company_resume_template.docx):
render(candidate: CandidateProfile, output_path: Path) -> Path — builds
the docxtpl context dict from the candidate's fields (dumping
work_history/education as dicts), renders, saves to output_path, creates
parent directories if needed, returns the path.

Add a POST /candidates/{candidate_id}/render-template endpoint that loads
the candidate's structured_json, renders it, and returns the file as a
downloadable response.

DO NOT start on the matching engine yet — that's Step 9.

Acceptance criteria:
- Calling the render endpoint for a processed candidate produces a valid
  .docx file that opens correctly and shows the candidate's data in the
  company template layout
```

---

### Step 8 — Embedding service with Postgres cache

```
Step 8 of 18. Template service from Step 7 exists.

Create services/embedding_service.py with EmbeddingService(model,
session_factory): embed_cached(key: str, text: str) -> np.ndarray —
checks embedding_cache table for the key; if present, returns the stored
vector; if not, computes model.encode(text, normalize_embeddings=True),
stores it in embedding_cache, and returns it.

Load the embedding model (BAAI/bge-small-en-v1.5 or equivalent, should
support Vietnamese, from sentence-transformers) once at app startup
and pass it into EmbeddingService via dependency injection — do not
reload the model per-request.

Do NOT add Qdrant, pgvector, or any vector database in this step — the
plan explicitly keeps candidate-pool matching as an in-memory numpy
operation. If you find yourself wanting a vector DB here, stop and
re-read why the implementation plan avoids one at this scale.

Acceptance criteria:
- Calling embed_cached twice with the same key returns the identical
  vector on the second call without a second call to model.encode
  (verify via a test with a mocked model whose encode() call count is
  asserted to be 1 after two embed_cached calls with the same key)
```

---

### Step 9 — Matching engine: hard filters

```
Step 9 of 18. Embedding service from Step 8 exists.

Create services/matching_service.py with MatchingService(embeddings,
llm, hard_filter_soft_cap=0.4). Implement ONLY apply_hard_filters in this
step:

apply_hard_filters(candidate, job) -> tuple[bool, str]:
- checks years_experience_total >= job.min_years_experience
- checks every job.must_have_skills item is present (case-insensitive) in
  candidate.skills
- checks candidate's highest education level (via an EDUCATION_RANK dict:
  high_school=0, associate=1, bachelor=2, master=3, phd=4) meets
  job.required_education_level if set
- checks every job.required_certifications item is present
  (case-insensitive) in candidate.certifications
- accumulates a human-readable note for each failing check; if all pass,
  note reads "Meets all hard requirements."
- never raises

Do NOT implement score_soft_fit or match() yet — those are Step 10.

Write tests/test_matching_service.py covering: missing must-have skill
fails with that skill named in the notes; insufficient years fails;
insufficient education level fails; all-requirements-met passes with the
"Meets all hard requirements" note. Use MatchingService(embeddings=None,
llm=None) since apply_hard_filters does no I/O.

Acceptance criteria:
- All four test cases pass
- apply_hard_filters never touches self._embeddings or self._llm
```

---

### Step 10 — Matching engine: soft-fit scoring + combination

```
Step 10 of 18. Hard filters from Step 9 exist.

Add to MatchingService:
- score_soft_fit(candidate, job) -> float: computes 6 cached embeddings
  (candidate skills/experience/education text, job skills/experience/
  education text) via self._embeddings.embed_cached with keys formatted
  as "candidate:{id}:skills" etc. and "job:{id}:skills" etc. Computes
  cosine similarity for each of the 3 pairs, combines with weights
  SKILL_WEIGHT=0.5, EXPERIENCE_WEIGHT=0.3, EDUCATION_WEIGHT=0.2.
- a module-level _cosine(a, b) helper handling the zero-vector edge case
  (return 0.0 if either norm is 0)
- match(candidate, job) -> MatchResult: calls apply_hard_filters and
  score_soft_fit, sets overall_score = soft_fit if passed else
  min(soft_fit, self._soft_cap), calls self._generate_explanation(...)
  for the explanation field, returns a populated MatchResult

Do NOT implement _generate_explanation's prompt yet — that's Step 11.
Stub it to return "" for now so match() runs end-to-end.

Acceptance criteria:
- Given a candidate and job with mocked embed_cached returning known
  vectors, score_soft_fit returns the expected weighted-average value
  (write a test with hand-computed expected output)
- match() returns a MatchResult with overall_score capped at 0.4 when
  hard filters fail, uncapped when they pass
```

---

### Step 11 — Match explanation + matches API endpoint

```
Step 11 of 18. Soft-fit scoring from Step 10 exists (with
_generate_explanation stubbed).

Create prompts/match_explanation_prompt.py with
build_match_explanation_prompt(candidate, job, passed, notes, soft_fit)
— a prompt instructing the LLM to write a 2-3 sentence explanation using
ONLY the candidate's skills/experience and the job's requirements listed
in the prompt, explicitly forbidding inventing any skill or credential
not listed.

Wire this into MatchingService._generate_explanation: build the prompt,
call self._llm.complete(prompt, max_tokens=200, temperature=0.2), return
the stripped response.

Create api/matches.py with POST /matches/run?job_id=...: loads the
job, loads every Candidate with extraction_status="ok", runs
matcher.match() for each, upserts results into the matches table, returns
the results sorted by overall_score descending. Wire into main.py.

Create api/jobs.py with POST /jobs: accepts {jd_text, title,
must_have_skills, nice_to_have_skills, min_years_experience,
required_education_level, required_certifications,
responsibilities_text}, inserts a Job row, returns {job_id}.

Acceptance criteria:
- POSTing a JD via /jobs then calling /matches/run?job_id=... against a
  pool of 5+ processed candidates returns a correctly ranked list with
  non-empty, plausible explanations for each
- Explanations do not mention any skill/credential absent from the
  candidate's structured_json (manually spot-check 3-4)
```

---

### Step 12 — Interview question generation

```
Step 12 of 18. Matching pipeline from Step 11 works end-to-end.

Create prompts/interview_questions_prompt.py with
build_interview_prompt(candidate, job) — instructs the LLM to return
exactly 5 questions as a JSON list: Q1-2 deep-diving the candidate's
strongest relevant experience, Q3-4 probing explicit gaps against the JD
(naming the gap), Q5 one role-specific behavioral question. Forbid
referencing anything not present in the given candidate/job JSON.

Create services/interview_service.py with InterviewService(llm):
generate_questions(candidate, job) -> list[str] — calls
llm.complete_structured(prompt), json.loads the result, logs a warning
(but does not raise) if the result isn't a 5-item list, returns [] on
JSONDecodeError.

Create api/interview.py with POST
/interview/generate?candidate_id=...&job_id=...: loads both records,
calls the service, stores the result in interview_questions, returns
{candidate_id, job_id, questions}. Wire into main.py.

Acceptance criteria:
- Calling the endpoint for a real matched candidate/job pair returns 5
  questions; manually verify at least one names a specific gap between
  the candidate's listed skills and the JD's must-haves
```

---

### Step 13 — Streamlit dashboard: upload + match tabs

```
Step 13 of 18. All API endpoints through Step 12 exist and work.

Create app/streamlit.py with three tabs: "Upload Resumes",
"Match Against JD", "Needs Review" (this step builds the first two; Step
14 builds the third).

Upload tab: multi-file uploader (pdf, docx), a Process button that POSTs
each file to /candidates/upload and displays the returned status.

Match tab: a text area for pasting a JD, a "Find Matches" button that
POSTs to /jobs then /matches/run, and renders each result in an expander
showing overall_score, hard-filter pass/fail with notes, and the
explanation text. Do not build the interview-question button yet — Step
14 finishes this tab.

Acceptance criteria:
- Running `streamlit run app/streamlit.py` against the live API
  lets you upload a batch of resumes and see their processing status
- Pasting a JD and clicking Find Matches shows a ranked, expandable list
  of real candidates with explanations
```

---

### Step 14 — Streamlit dashboard: interview generation + review tab

```
Step 14 of 18. Upload/match tabs from Step 13 exist.

In the Match tab, add a "Generate interview questions" button inside each
candidate's expander that calls /interview/generate for that
candidate_id/job_id and renders the returned 5 questions as a numbered
list.

Build the "Needs Review" tab: fetch candidates with
extraction_status="needs_review" (add a GET /candidates?status=X query
param to api/candidates.py if not already supported), display the
original uploaded file alongside whatever partial structured_json exists,
with a simple editable form (text inputs bound to each CandidateProfile
field) and a "Save & Mark OK" button that PATCHes the candidate record
directly (add PATCH /candidates/{id} if needed) and sets
extraction_status="ok".

Acceptance criteria:
- Generating interview questions from the dashboard displays 5 questions
  inline without a page reload
- A candidate deliberately forced into needs_review (e.g. by feeding a
  corrupted file) appears in the Review tab and can be manually corrected
  and marked ok
```

---

### Step 15 — Labeled evaluation set

```
Step 15 of 18. Full pipeline through Step 14 works end-to-end.

Create eval/labeled_set.jsonl with 5-8 entries, each a JSON object:
{"job_id": "...", "human_ranking": ["candidate_id_1", "candidate_id_2", ...],
 "hard_filter_labels": [{"candidate_id": "...", "human_passed": true}, ...]}
Populate this using 5-8 real or realistic JDs, each matched against
10-15 candidates already processed through the pipeline, with a human
(you) ranking them and labeling hard-filter pass/fail by hand.

Do NOT write run_eval.py yet — that's Step 16. This step is just building
the labeled dataset.

Acceptance criteria:
- labeled_set.jsonl has 5-8 well-formed entries
- Every candidate_id referenced actually exists in the candidates table
  with extraction_status="ok"
```

---

### Step 16 — Evaluation harness

```
Step 16 of 18. Labeled set from Step 15 exists.

Create eval/run_eval.py: load_labeled_set(path) reads the JSONL. Run
/matches/run for every job_id in the labeled set and collect results.
evaluate(labeled_set, system_results) computes:
- mean Spearman rank correlation between human_ranking and the system's
  ranking (only over candidates present in both lists)
- hard-filter precision and recall against human_passed labels (compare
  against system's passed_hard_filters for the same candidate_id/job_id)

Print a summary report: mean_ranking_spearman, hard_filter_precision,
hard_filter_recall.

Do NOT build a CI gate on these metrics (unlike Projects 1 & 2's eval-gate
extended-scope item) — that was scoped to those two projects specifically,
not a default to reuse here.

Acceptance criteria:
- Running `python eval/run_eval.py` against the labeled set and live
  pipeline prints all three metrics without error
- Spot-check: manually verify the Spearman calculation against 2-3 known
  rank pairs by hand
```

---

### Step 17 — Test suite + CI

```
Step 17 of 18. Eval harness from Step 16 exists.

Ensure tests/test_matching_service.py, tests/test_extraction_service.py,
tests/test_pii_scrub.py, and tests/test_candidate_schema.py (from earlier
steps) all exist and pass together via `pytest tests/ -v`.

Create .github/workflows/ci.yml: on push/PR, checkout, setup Python 3.12,
pip install -r requirements.txt, ruff check ., pytest tests/ -v. No eval
gate in this workflow.

Acceptance criteria:
- `pytest tests/ -v` passes locally with zero failures and zero live
  network calls (everything LLM-related is mocked)
- Pushing to a test branch triggers the GitHub Actions workflow
  successfully
```

---

### Step 18 — README + polish

```
Step 18 of 18. Everything through Step 17 works.

Write README.md covering: project overview and one-line pitch, why no
job queue and no vector DB are used at this scale (with a note that both
are documented, ready swap-ins if volume grew), setup instructions
(docker compose up, alembic upgrade head, streamlit run), an architecture
diagram (ASCII is fine), and a "Design Decisions & PII/Fairness" section
that explicitly lists the excluded protected-characteristic fields, states
the tool is decision-support only (no auto-reject/auto-advance), and
explains the hard-filter-then-soft-fit matching approach versus a full
hybrid retrieval stack.

Take 3-4 demo screenshots (upload flow, ranked match list with an
explanation visible, generated interview questions, the review tab) and
reference them in the README.

Acceptance criteria:
- A reviewer with no prior context can clone the repo, follow the README,
  and get a working demo running end-to-end without asking you a
  clarifying question
```
