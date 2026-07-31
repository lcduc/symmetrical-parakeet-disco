"""HR CV Matcher dashboard.

Talks to the FastAPI backend over HTTP only — no direct DB or service
imports — so it can be pointed at any deployment via API_BASE_URL.
"""

import logging
import os
import secrets
from urllib.parse import unquote

import fitz
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

API_BASE_URL = os.environ.get("API_BASE_URL") or f"http://localhost:{os.environ.get('API_PORT', '8002')}"
APP_API_KEY = os.environ.get("APP_API_KEY", "")
REQUEST_TIMEOUT_SECONDS = 60
EDUCATION_LEVEL_OPTIONS = ["", "high_school", "associate", "bachelor", "master", "phd"]


def _api_headers() -> dict[str, str]:
    """Headers to attach to every backend request; carries the shared API key, if configured."""
    return {"X-API-Key": APP_API_KEY} if APP_API_KEY else {}


def _require_passphrase() -> bool:
    """Gate the whole dashboard behind APP_API_KEY, entered once per browser session.

    Returns True once the correct passphrase has been entered (or no
    APP_API_KEY is configured, e.g. local demo use).
    """
    if not APP_API_KEY:
        return True
    if st.session_state.get("authenticated"):
        return True

    st.title("HR CV Matcher")
    entered = st.text_input("Access passphrase", type="password", key="passphrase_input")
    if st.button("Unlock"):
        if secrets.compare_digest(entered, APP_API_KEY):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect passphrase.")
    return False


# ------------------------------------------------------------------------------
# Cached API Fetch Helpers
# ------------------------------------------------------------------------------

@st.cache_data(ttl=30)
def _fetch_candidates(status: str | None = None) -> list[dict]:
    """GET /candidates, optionally filtered by extraction_status. Returns empty list on failure."""
    try:
        params = {"status": status} if status else {}
        response = requests.get(
            f"{API_BASE_URL}/candidates", params=params, headers=_api_headers(), timeout=REQUEST_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        logger.exception("Failed to fetch candidates (status=%s)", status)
        st.error("Failed to fetch candidates.")
        return []


@st.cache_data(ttl=60)
def _fetch_candidate(candidate_id: str) -> dict | None:
    """GET /candidates/{id}, or None on failure."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/candidates/{candidate_id}", headers=_api_headers(), timeout=REQUEST_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        logger.exception("Failed to fetch candidate %s", candidate_id)
        st.error("Failed to fetch candidate.")
        return None


# ------------------------------------------------------------------------------
# Utility Helpers
# ------------------------------------------------------------------------------

def _calculate_textarea_height(text: str, default_height: int = 68) -> int:
    """Calculate height for text areas to fit ~2 rows by default, expanding only when needed."""
    if not text:
        return default_height

    lines = text.splitlines()
    total_lines = 0
    for line in lines:
        total_lines += max(1, (len(line) + 69) // 70)

    if total_lines <= 2:
        return default_height

    return default_height + (total_lines - 2) * 20


def _infer_education_level(level_val: str | None, degree_val: str | None) -> str:
    """Resolve level from entry['level'] or infer matching option from entry['degree']."""
    if level_val:
        normalized = str(level_val).strip().lower().replace(" ", "_")
        if normalized in EDUCATION_LEVEL_OPTIONS:
            return normalized

    if not degree_val:
        return ""

    deg = str(degree_val).lower()
    if "phd" in deg or "doctor" in deg:
        return "phd"
    if "master" in deg or "m.s" in deg or "m.a" in deg or "mba" in deg:
        return "master"
    if "bachelor" in deg or "b.s" in deg or "b.a" in deg or "bs" in deg or "ba" in deg:
        return "bachelor"
    if "associate" in deg or "a.s" in deg or "a.a" in deg:
        return "associate"
    if "high school" in deg or "secondary" in deg:
        return "high_school"

    return ""


def _parse_skill_list(raw_text: str) -> list[str]:
    """Split comma or newline separated skills/certifications into a clean list."""
    cleaned = []
    for line in raw_text.splitlines():
        for item in line.split(","):
            if item.strip():
                cleaned.append(item.strip())
    return cleaned


def _parse_content_disposition_filename(header: str, fallback: str) -> str:
    """Extract the filename from a Content-Disposition header.

    Prefers the RFC 5987/6266 `filename*=<encoding>''<percent-encoded>` form
    (what Starlette's FileResponse emits whenever the filename has a space or
    any non-ASCII character, which is effectively every real name) over the
    plain `filename="..."` form, since the extended form is the only one
    guaranteed to be present in that case.
    """
    if not header:
        return fallback

    for part in header.split(";"):
        part = part.strip()
        if part.lower().startswith("filename*="):
            value = part.split("=", 1)[1].strip()
            encoding, _, encoded = value.partition("''")
            try:
                return unquote(encoded, encoding=encoding or "utf-8")
            except LookupError:
                return unquote(encoded)

    for part in header.split(";"):
        part = part.strip()
        if part.lower().startswith("filename="):
            return part.split("=", 1)[1].strip().strip('"')

    return fallback


def _extract_error_detail(exc: requests.RequestException, fallback_message: str) -> str:
    """Extract FastAPI detail message from exception if available."""
    if exc.response is not None:
        try:
            detail = exc.response.json().get("detail")
            if detail:
                if isinstance(detail, list):
                    messages = [f"{err.get('loc', [])[-1]}: {err.get('msg')}" for err in detail if isinstance(err, dict)]
                    return f"Validation error: {', '.join(messages)}"
                return str(detail)
        except ValueError:
            logger.exception("Failed to parse error response body as JSON")
    return fallback_message


def _extract_text_from_jd_file(uploaded_file) -> str:
    """Extract raw text from uploaded JD file (PDF, DOCX, TXT)."""
    file_type = uploaded_file.name.split(".")[-1].lower()
    content = uploaded_file.getvalue()

    if file_type == "pdf":
        try:
            text = ""
            with fitz.open(stream=content, filetype="pdf") as doc:
                for page in doc:
                    text += page.get_text() + "\n"
            return text.strip()
        except Exception:
            logger.exception("Failed to parse PDF JD file")
            return ""
    elif file_type == "txt":
        try:
            return content.decode("utf-8", errors="ignore").strip()
        except UnicodeDecodeError:
            logger.exception("Failed to decode TXT JD file")
            return ""
    return ""


# ------------------------------------------------------------------------------
# API Mutation Helpers
# ------------------------------------------------------------------------------

def _create_job(job_payload: dict) -> str | None:
    """POST job requirements to /jobs and return job_id, or None on failure."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/jobs", json=job_payload, headers=_api_headers(), timeout=REQUEST_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        return response.json()["job_id"]
    except requests.RequestException as exc:
        logger.exception("Failed to create job")
        st.error(_extract_error_detail(exc, "Failed to create job from the JD."))
        return None


def _run_matches(job_id: str) -> list[dict] | None:
    """POST to /matches/run and return ranked match list, or None on failure."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/matches/run",
            params={"job_id": job_id},
            headers=_api_headers(),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        logger.exception("Failed to run matches for job %s", job_id)
        st.error(_extract_error_detail(exc, "Failed to run matching."))
        return None


def _generate_interview_questions(candidate_id: str, job_id: str) -> list[str] | None:
    """POST to /interview/generate and return interview questions, or None on failure."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/interview/generate",
            params={"candidate_id": candidate_id, "job_id": job_id},
            headers=_api_headers(),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()["questions"]
    except requests.RequestException as exc:
        logger.exception("Failed to generate interview questions for candidate %s / job %s", candidate_id, job_id)
        st.error(_extract_error_detail(exc, "Failed to generate interview questions."))
        return None


def _save_candidate_correction(candidate_id: str, patch_payload: dict) -> bool:
    """PATCH /candidates/{id} with corrected fields. Returns True on success."""
    try:
        response = requests.patch(
            f"{API_BASE_URL}/candidates/{candidate_id}",
            json=patch_payload,
            headers=_api_headers(),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        st.cache_data.clear()
        return True
    except requests.RequestException as exc:
        logger.exception("Failed to save correction for candidate %s", candidate_id)
        st.error(_extract_error_detail(exc, f"Failed to save candidate {candidate_id}."))
        return False


def _fetch_company_template_bytes(candidate_id: str) -> tuple[bytes, str] | None:
    """POST /candidates/{id}/render-template and return (content, filename) or None on error."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/candidates/{candidate_id}/render-template",
            headers=_api_headers(),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.exception("Failed to render company template for candidate %s", candidate_id)
        st.error(_extract_error_detail(exc, "Failed to export candidate to company template."))
        return None

    content_disposition = response.headers.get("content-disposition", "")
    filename = _parse_content_disposition_filename(content_disposition, fallback=f"{candidate_id}.pdf")
    return response.content, filename


# ------------------------------------------------------------------------------
# Sidebar Component Renderers
# ------------------------------------------------------------------------------

@st.cache_data(ttl=60)
def _fetch_candidate_file_bytes(candidate_id: str) -> tuple[bytes, str] | None:
    """GET /candidates/{id}/file, cached so re-renders don't re-download on every rerun."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/candidates/{candidate_id}/file", headers=_api_headers(), timeout=REQUEST_TIMEOUT_SECONDS
        )
        response.raise_for_status()
    except requests.RequestException:
        logger.exception("Failed to fetch original file for candidate %s", candidate_id)
        return None
    return response.content, response.headers.get("content-type", "")


def _render_file_preview(candidate_id: str) -> None:
    """Render a crisp first-page preview (PDF only) of the candidate's original file."""
    fetched = _fetch_candidate_file_bytes(candidate_id)
    if fetched is None:
        st.sidebar.warning("Original file unavailable.")
        return
    content, content_type = fetched

    if content_type != "application/pdf":
        st.sidebar.caption("Page preview is only available for PDF files.")
        return

    try:
        with fitz.open(stream=content, filetype="pdf") as document:
            pixmap = document[0].get_pixmap(matrix=fitz.Matrix(2, 2))
            page_image = pixmap.tobytes("png")
        st.sidebar.image(page_image, caption="Page 1", width="stretch")
    except Exception:
        logger.exception("Failed to render preview image for candidate %s", candidate_id)
        st.sidebar.warning("Could not render a page preview for this file.")


def render_resume_sidebar() -> None:
    """Sidebar view for Resume Upload & Preview (Candidate Detail & Review views)."""
    st.sidebar.header("Upload Resumes")
    uploaded_files = st.sidebar.file_uploader(
        "Select resumes (PDF or DOCX)",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        key="resume_uploader",
    )

    if st.sidebar.button("Process Resumes", disabled=not uploaded_files, key="process_resumes_btn"):
        uploaded_any = False
        progress_bar = st.sidebar.progress(0.0)
        for index, uploaded_file in enumerate(uploaded_files):
            with st.sidebar.status(f"Uploading {uploaded_file.name}...", expanded=False):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/candidates/upload",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue())},
                        headers=_api_headers(),
                        timeout=REQUEST_TIMEOUT_SECONDS,
                    )
                    response.raise_for_status()
                    body = response.json()
                    st.sidebar.success(f"{uploaded_file.name} -> status={body['status']}")
                    uploaded_any = True
                except requests.RequestException as exc:
                    logger.exception("Upload failed for %s", uploaded_file.name)
                    st.sidebar.error(f"{uploaded_file.name}: {_extract_error_detail(exc, 'Upload failed')}")
            progress_bar.progress((index + 1) / len(uploaded_files))

        if uploaded_any:
            st.cache_data.clear()

    selected_candidate_id = st.session_state.get("selected_candidate_id")
    if selected_candidate_id:
        st.sidebar.divider()
        st.sidebar.subheader("Resume Preview")
        _render_file_preview(selected_candidate_id)


def render_jd_sidebar() -> None:
    """Dedicated sidebar view for Uploading Job Descriptions (Match Against JD view)."""
    st.sidebar.header("Upload Job Description")
    uploaded_jd = st.sidebar.file_uploader(
        "Select JD file (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=False,
        key="jd_uploader",
    )

    if st.sidebar.button("Process JD", disabled=not uploaded_jd, key="process_jd_btn"):
        extracted_text = _extract_text_from_jd_file(uploaded_jd)
        if extracted_text:
            st.session_state["jd_text_input"] = extracted_text
            if not st.session_state.get("jd_title_input"):
                raw_title = os.path.splitext(uploaded_jd.name)[0].replace("_", " ").replace("-", " ").title()
                st.session_state["jd_title_input"] = raw_title
            st.sidebar.success("JD text extracted successfully!")
            st.rerun()
        else:
            st.sidebar.error("Could not extract text from uploaded JD file.")


# ------------------------------------------------------------------------------
# Form Field Renderers
# ------------------------------------------------------------------------------

def _render_work_history_entry(entry: dict, widget_key: str) -> dict:
    """Render one work-history entry as separate input boxes and return edited entry."""
    col1, col2 = st.columns(2)
    with col1:
        company = st.text_input("Company", value=entry.get("company") or "", key=f"{widget_key}_company")
    with col2:
        title = st.text_input("Title", value=entry.get("title") or "", key=f"{widget_key}_title")

    date_col1, date_col2 = st.columns(2)
    with date_col1:
        start_date = st.text_input("Start date", value=entry.get("start_date") or "", key=f"{widget_key}_start")
    with date_col2:
        end_date = st.text_input("End date", value=entry.get("end_date") or "", key=f"{widget_key}_end")
    
    desc_list = entry.get("description") or []
    desc_text = "\n".join(desc_list) if isinstance(desc_list, list) else str(desc_list)
    
    description = st.text_area(
        "Description bullets (one per line)",
        value=desc_text,
        key=f"{widget_key}_description",
        height="content",
    )
    return {
        "company": company,
        "title": title,
        "start_date": start_date,
        "end_date": end_date or None,
        "description": [line.strip() for line in description.splitlines() if line.strip()],
    }


def _render_education_entry(entry: dict, widget_key: str) -> dict:
    """Render one education entry as separate input boxes and return edited entry."""
    degree_val = entry.get("degree") or ""
    level_val = entry.get("level") or ""

    resolved_level = _infer_education_level(level_val, degree_val)
    level_index = (
        EDUCATION_LEVEL_OPTIONS.index(resolved_level)
        if resolved_level in EDUCATION_LEVEL_OPTIONS
        else 0
    )

    col1, col2 = st.columns(2)
    with col1:
        institution = st.text_input("Institution", value=entry.get("institution") or "", key=f"{widget_key}_institution")
    with col2:
        level = st.selectbox(
            "Level",
            options=EDUCATION_LEVEL_OPTIONS,
            index=level_index,
            format_func=lambda value: "None" if value == "" else value,
            key=f"{widget_key}_level",
        )

    col3, col4 = st.columns(2)
    with col3:
        degree = st.text_input("Degree", value=degree_val, key=f"{widget_key}_degree")
    with col4:
        field_of_study = st.text_input(
            "Field of study", value=entry.get("field_of_study") or "", key=f"{widget_key}_field"
        )

    date_col1, date_col2 = st.columns(2)
    with date_col1:
        start_date = st.text_input("Start date", value=entry.get("start_date") or "", key=f"{widget_key}_start")
    with date_col2:
        end_date = st.text_input("End date", value=entry.get("end_date") or "", key=f"{widget_key}_end")
    
    desc_list = entry.get("description") or []
    desc_text = "\n".join(desc_list) if isinstance(desc_list, list) else str(desc_list)

    description = st.text_area(
        "Description bullets (one per line)",
        value=desc_text,
        key=f"{widget_key}_description",
        height="content",
    )
    return {
        "institution": institution or None,
        "degree": degree or None,
        "field_of_study": field_of_study or None,
        "level": level or None,
        "start_date": start_date or None,
        "end_date": end_date or None,
        "description": [line.strip() for line in description.splitlines() if line.strip()],
    }


def render_candidate_editor(candidate: dict, key_prefix: str) -> None:
    """Editable form for candidate's parsed profile."""
    candidate_id = candidate["candidate_id"]
    candidate_status = candidate.get("status", "")
    profile = candidate.get("profile") or {}

    existing_work_history = profile.get("work_history") or []
    existing_education = profile.get("education") or []

    work_history_count_key = f"{key_prefix}_wh_count_{candidate_id}"
    education_count_key = f"{key_prefix}_edu_count_{candidate_id}"

    if work_history_count_key not in st.session_state:
        st.session_state[work_history_count_key] = len(existing_work_history)
    if education_count_key not in st.session_state:
        st.session_state[education_count_key] = len(existing_education)

    work_history_count = st.session_state[work_history_count_key]
    education_count = st.session_state[education_count_key]

    # No st.form here: the export button below needs to read whatever is
    # currently typed in these fields, and widgets inside a form only sync
    # their values to the script on submit — so the fields must be plain,
    # immediately-synced widgets for a standalone button to see live edits.
    fn_col1, fn_col2 = st.columns(2)
    with fn_col1:
        full_name = st.text_input(
            "Full name", value=profile.get("full_name") or "", key=f"{key_prefix}_full_name_{candidate_id}"
        )
    with fn_col2:
        st.text_input("Status", value=candidate_status, disabled=True, key=f"{key_prefix}_status_{candidate_id}")

    ep_col1, ep_col2 = st.columns(2)
    with ep_col1:
        email = st.text_input("Email", value=profile.get("email") or "", key=f"{key_prefix}_email_{candidate_id}")
    with ep_col2:
        phone = st.text_input("Phone", value=profile.get("phone") or "", key=f"{key_prefix}_phone_{candidate_id}")

    # --- Work History Section ---
    st.write("**Work history**")
    work_history = []
    for index in range(work_history_count):
        entry = existing_work_history[index] if index < len(existing_work_history) else {}
        st.caption(f"Entry {index + 1}")
        work_history.append(_render_work_history_entry(entry, f"{key_prefix}_wh_{candidate_id}_{index}"))

    wh_col1, wh_col2, _ = st.columns([1, 1, 3])
    with wh_col1:
        add_wh = st.button("➕ Add Work Entry", key=f"{key_prefix}_btn_add_wh_{candidate_id}")
    with wh_col2:
        rem_wh = st.button("➖ Remove Work Entry", key=f"{key_prefix}_btn_rem_wh_{candidate_id}")

    st.divider()

    # --- Education Section ---
    st.write("**Education**")
    education = []
    for index in range(education_count):
        entry = existing_education[index] if index < len(existing_education) else {}
        st.caption(f"Entry {index + 1}")
        education.append(_render_education_entry(entry, f"{key_prefix}_edu_{candidate_id}_{index}"))

    edu_col1, edu_col2, _ = st.columns([1, 1, 3])
    with edu_col1:
        add_edu = st.button("➕ Add Education Entry", key=f"{key_prefix}_btn_add_edu_{candidate_id}")
    with edu_col2:
        rem_edu = st.button("➖ Remove Education Entry", key=f"{key_prefix}_btn_rem_edu_{candidate_id}")

    st.divider()

    raw_years = profile.get("years_experience_total")
    try:
        default_years = float(raw_years) if raw_years is not None else 0.0
    except (ValueError, TypeError):
        default_years = 0.0

    yoe_col1, _ = st.columns(2)
    with yoe_col1:
        years_experience_total = st.number_input(
            "Years of experience",
            min_value=0.0,
            step=0.5,
            value=default_years,
            key=f"{key_prefix}_yoe_{candidate_id}",
        )

    skills_text = ", ".join(profile.get("skills") or [])
    skills = st.text_area(
        "Skills (comma-separated)",
        value=skills_text,
        key=f"{key_prefix}_skills_{candidate_id}",
        height="content",
    )

    certs_text = ", ".join(profile.get("certifications") or [])
    certifications = st.text_area(
        "Certifications (comma-separated)",
        value=certs_text,
        key=f"{key_prefix}_certifications_{candidate_id}",
        height="content",
    )

    st.divider()

    if add_wh:
        st.session_state[work_history_count_key] += 1
        st.rerun()
    elif rem_wh and st.session_state[work_history_count_key] > 0:
        st.session_state[work_history_count_key] -= 1
        st.rerun()
    elif add_edu:
        st.session_state[education_count_key] += 1
        st.rerun()
    elif rem_edu and st.session_state[education_count_key] > 0:
        st.session_state[education_count_key] -= 1
        st.rerun()

    # A single button slot that toggles behavior: plain button while no export
    # is ready (clicking it saves + generates the file), then it becomes the
    # actual download_button once the file is ready. Downloading clears the
    # cached file so the next click regenerates it from the current fields
    # instead of silently re-serving a stale export.
    export_file_key = f"export_file_{candidate_id}"
    export_file = st.session_state.get(export_file_key)
    export_button_key = f"{key_prefix}_btn_export_{candidate_id}"

    if export_file:
        downloaded = st.download_button(
            "Save & export",
            data=export_file["content"],
            file_name=export_file["filename"],
            mime="application/pdf",
            type="primary",
            key=export_button_key,
        )
        if downloaded:
            del st.session_state[export_file_key]
    elif st.button("Save & export", type="primary", key=export_button_key):
        patch_payload = {
            "full_name": full_name,
            "email": email,
            "phone": phone or None,
            "skills": _parse_skill_list(skills),
            "certifications": _parse_skill_list(certifications),
            "years_experience_total": years_experience_total,
            "work_history": work_history,
            "education": education,
        }
        if _save_candidate_correction(candidate_id, patch_payload):
            st.success(f"Candidate {candidate_id} saved.")
            result = _fetch_company_template_bytes(candidate_id)
            if result:
                content, filename = result
                st.session_state[export_file_key] = {"content": content, "filename": filename}
                st.rerun()


# ------------------------------------------------------------------------------
# Dashboard Views
# ------------------------------------------------------------------------------

def render_candidate_detail_tab() -> None:
    """Detailed, editable view with Candidate Selector at top."""
    header_col, refresh_col = st.columns([5, 1])
    with header_col:
        st.header("Candidate Detail")
    with refresh_col:
        if st.button("Refresh", help="Check for candidates that finished background processing"):
            st.cache_data.clear()
            st.rerun()

    candidates = _fetch_candidates()
    if not candidates:
        st.info("No candidates uploaded yet.")
        st.session_state["selected_candidate_id"] = None
        return

    def _label(candidate: dict) -> str:
        name = (candidate.get("profile") or {}).get("full_name") or candidate["candidate_id"]
        return f"{name} ({candidate['status']})"

    labels_to_ids = {_label(candidate): candidate["candidate_id"] for candidate in candidates}
    
    default_index = 0
    current_id = st.session_state.get("selected_candidate_id")
    if current_id:
        for idx, cid in enumerate(labels_to_ids.values()):
            if cid == current_id:
                default_index = idx
                break

    selected_label = st.selectbox("Select Candidate", list(labels_to_ids.keys()), index=default_index)
    selected_candidate_id = labels_to_ids[selected_label]
    st.session_state["selected_candidate_id"] = selected_candidate_id

    st.divider()

    candidate = _fetch_candidate(selected_candidate_id)
    if candidate is None:
        return

    render_candidate_editor(candidate, key_prefix="detail")


def render_match_tab() -> None:
    """JD input + structured requirements, then ranked, expandable match results."""
    st.header("Match Against JD")

    title = st.text_input("Job title", key="jd_title_input")
    jd_text = st.text_area("Job description", height="content", key="jd_text_input")

    col1, col2 = st.columns(2)
    with col1:
        min_years_experience = st.number_input("Minimum years experience", min_value=0.0, step=0.5)
    with col2:
        required_education_level = st.selectbox(
            "Required education level",
            options=[None, "high_school", "associate", "bachelor", "master", "phd"],
            format_func=lambda value: "None" if value is None else value,
        )

    if st.button("Find Matches", disabled=not (title and jd_text)):
        job_payload = {
            "title": title,
            "jd_text": jd_text,
            "min_years_experience": min_years_experience,
            "required_education_level": required_education_level,
            "responsibilities_text": "",
        }
        job_id = _create_job(job_payload)
        if job_id is None:
            return

        matches = _run_matches(job_id)
        if matches is None:
            return

        st.session_state["job_id"] = job_id
        st.session_state["matches"] = matches

    matches = st.session_state.get("matches", [])
    if not matches:
        return

    interview_questions = st.session_state.setdefault("interview_questions", {})
    expanded_candidates = st.session_state.setdefault("expanded_candidates", set())

    st.subheader(f"Ranked candidates ({len(matches)})")
    for match in matches:
        candidate_id = match["candidate_id"]
        candidate_name = match.get("candidate_name") or candidate_id
        filter_icon = "✅" if match["passed_hard_filters"] else "❌"
        flag_icon = " ⚠️" if match.get("experience_flag") else ""
        overall_pct = match["overall_score"] * 100
        label = f"{filter_icon}{flag_icon} {candidate_name} — {overall_pct:.0f}% match"
        
        is_expanded = candidate_id in expanded_candidates or candidate_id in interview_questions
        with st.expander(label, expanded=is_expanded):
            st.metric("Overall match", f"{overall_pct:.0f}%")
            st.write(f"**Hard filters:** {'Passed' if match['passed_hard_filters'] else 'Failed'}")
            st.write(match["hard_filter_notes"])
            if match.get("experience_flag"):
                st.warning(match["experience_flag"])
            if not match["passed_hard_filters"]:
                st.caption(f"Raw experience-similarity score (uncapped): {match['soft_fit_score'] * 100:.0f}%")
            st.write("**Explanation**")
            st.write(match["explanation"])

            if st.button("Generate interview questions", key=f"generate_interview_{candidate_id}"):
                expanded_candidates.add(candidate_id)
                questions = _generate_interview_questions(candidate_id, match["job_id"])
                if questions is not None:
                    interview_questions[candidate_id] = questions
                    st.rerun()

            if candidate_id in interview_questions:
                st.write("**Interview questions**")
                for index, question in enumerate(interview_questions[candidate_id], start=1):
                    st.write(f"{index}. {question}")


def render_review_tab() -> None:
    """List candidates needing manual review with editable correction forms."""
    st.header("Needs Review")
    candidates = _fetch_candidates(status="needs_review")

    if not candidates:
        st.info("No candidates currently need review.")
        return

    st.write(f"{len(candidates)} candidate(s) awaiting review.")
    for candidate in candidates:
        with st.expander(candidate["candidate_id"]):
            render_candidate_editor(candidate, key_prefix="review")


# ------------------------------------------------------------------------------
# Main Entry Point with Dynamic Sidebar Navigation
# ------------------------------------------------------------------------------

def main() -> None:
    """Render HR CV Matcher dashboard with context-aware sidebars."""
    st.set_page_config(page_title="HR CV Matcher", layout="wide")

    if not _require_passphrase():
        return

    st.title("HR CV Matcher")

    selected_view = st.radio(
        "Navigation",
        options=["Candidate Detail", "Match Against JD", "Needs Review"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if selected_view == "Match Against JD":
        render_jd_sidebar()
        render_match_tab()
    elif selected_view == "Needs Review":
        render_review_tab()
    else:  # "Candidate Detail"
        render_resume_sidebar()
        render_candidate_detail_tab()


if __name__ == "__main__":
    main()