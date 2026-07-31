import shutil
from pathlib import Path

import pytest

from models.candidate import CandidateProfile, EducationEntry, WorkExperience
from services.template_service import TemplateService

PREVIEW_OUTPUT_PATH = Path("data/rendered/template_preview.pdf")

pytestmark = pytest.mark.skipif(shutil.which("tectonic") is None, reason="tectonic not on PATH")


def _preview_profile() -> CandidateProfile:
    """A fixed, fully-populated profile for iterating on the template's layout.

    Exercises every optional branch the template renders differently:
    an ongoing job (end_date="Present"), a finished job, an ongoing degree,
    multi-bullet descriptions, open-source projects, research publications,
    certifications, and auxiliary metadata.
    """
    return CandidateProfile(
        full_name="Jamie Rivera",
        email="jamie.rivera@example.com",
        phone="+1 555-867-5309",
        location="San Francisco, CA",
        linkedin="linkedin.com/in/jamie-rivera",
        github="github.com/jamierivera",
        website="https://jamierivera.dev",
        summary=(
            "Senior AI Engineer with 4.5+ years of experience designing and scaling "
            "production RAG pipelines, fine-tuning LLMs, and building high-throughput ML microservices. "
            "Track record of reducing inference costs and latency while leading cross-functional teams."
        ),
        skills=[
            "Python",
            "FastAPI",
            "PostgreSQL",
            "Docker",
            "LangChain",
            "PyTorch",
            "Kubernetes",
            "vLLM",
            "Qdrant",
            "Triton Inference Server",
            "Redis",
            "Git / GitHub Actions",
        ],
        work_history=[
            WorkExperience(
                company="Nimbus Analytics",
                title="Senior AI Engineer",
                start_date="Jan 2023",
                end_date="Present",
                location="San Francisco, CA",
                description=[
                    "Led design and rollout of a RAG-based support assistant handling 10k+ daily queries with 94% satisfaction score.",
                    "Cut inference latency 40% and infrastructure cost 25% by migrating batch jobs to a streaming pipeline using vLLM and Redis.",
                    "Mentored two junior engineers through their first production ML deployments, establishing automated regression testing CI/CD pipelines.",
                    "Architected semantic search retrieval across 2M+ unstructured PDF documents using Qdrant vector database and custom hybrid ranking.",
                ],
            ),
            WorkExperience(
                company="Fernwood Labs",
                title="Machine Learning Engineer",
                start_date="Jun 2020",
                end_date="Dec 2022",
                location="Boston, MA",
                description=[
                    "Built and shipped a document-classification service deployed via Docker & Kubernetes across 3 core product lines.",
                    "Automated the model retraining pipeline using Airflow, reducing manual retraining cycle time from 3 days to under 2 hours.",
                    "Optimized Transformer inference times by 2.5x through ONNX Runtime quantization and TensorRT engine serialization.",
                ],
            ),
            WorkExperience(
                company="Apex Software",
                title="Software Engineering Intern",
                start_date="May 2019",
                end_date="Aug 2019",
                location="Seattle, WA",
                description=[
                    "Developed backend REST API endpoints using FastAPI and PostgreSQL to ingest real-time IoT telemetry data.",
                    "Created automated integration test suites achieving 88% line coverage across data processing modules.",
                ],
            ),
        ],
        education=[
            EducationEntry(
                institution="Northbridge University",
                degree="M.Sc.",
                field_of_study="Computer Science",
                level="Master",
                start_date="Sep 2024",
                end_date="Present",
                location="Cambridge, MA",
                description=["Thesis work on retrieval-augmented generation (RAG) fine-tuning for low-resource languages."],
            ),
            EducationEntry(
                institution="Northbridge University",
                degree="B.Sc.",
                field_of_study="Computer Science",
                level="Bachelor",
                start_date="Sep 2016",
                end_date="May 2020",
                location="Cambridge, MA",
                description=["Graduated Magna Cum Laude. Minor in Applied Mathematics."],
            ),
        ],
        projects=[
            {
                "name": "DocuParse-LLM",
                "role": "Creator & Maintainer",
                "url": "github.com/jamierivera/docuparse-llm",
                "description": "Open-source Python library for extract-and-parse pipelines targeting complex PDF tables using OCR + Vision LLMs (1.2k Stars on GitHub).",
            },
            {
                "name": "VectorCache",
                "role": "Contributor",
                "url": "github.com/jamierivera/vector-cache",
                "description": "Lightweight semantic caching layer for OpenAI/Anthropic API calls using cosine similarity thresholds.",
            },
        ],
        publications=[
            "Rivera, J., & Chen, L. (2025). Efficient Chunking Strategies for Dense Retrieval in Domain-Specific RAG. *Proceedings of EMNLP 2025*.",
        ],
        certifications=[
            "AWS Certified Machine Learning – Specialty",
            "Certified Kubernetes Administrator (CKA)",
            "Deep Learning Specialization – DeepLearning.AI",
        ],
        awards=[
            "Nimbus Analytics Tech Innovation Award (Q3 2024)",
            "Northbridge CS Department Outstanding Senior Thesis (2020)",
        ],
        languages=["English (Native)", "Spanish (Professional Working)"],
        years_experience_total=4.5,
    )

def test_render_template_preview():
    """Not a correctness assertion beyond "it compiles" — run this and open PREVIEW_OUTPUT_PATH.

    Reruns instantly on every template edit since it skips ingestion and
    the LLM extraction step entirely, using a fixed in-memory profile.
    """
    profile = _preview_profile()

    PREVIEW_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_path = TemplateService(tectonic_path="tectonic").render(profile, PREVIEW_OUTPUT_PATH)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
