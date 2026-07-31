"""One-off seeding script for the Step 15 labeled evaluation set.

Runs a pool of synthetic resumes through the real extraction pipeline
(ExtractionService + the configured LLM adapter) and creates a handful of
job postings, so eval/labeled_set.jsonl can reference real candidates/jobs
with extraction_status="ok" instead of fabricated IDs.

Usage: python -m eval.seed_eval_data
"""

import asyncio
import logging

from db import async_session_factory
from models.orm_models import Candidate, Job
from services.extraction_service import ExtractionService
from services.llm_adapter import build_llm_adapter

logger = logging.getLogger(__name__)

RESUMES: dict[str, str] = {
    "cv_senior_strong": """
Nguyen Van An
Email: an.nguyen.cv@example.com | Phone: +84 901 111 222

Summary: Computer Vision Engineer with 4 years of experience building
production object detection and segmentation systems.

Work Experience:
Senior Computer Vision Engineer, VisionTech AI (2022 - Present)
- Trained and deployed YOLOv8-based object detection models for retail
  shelf analytics, achieving 92% mAP on production data.
- Built a video analytics pipeline processing 200+ camera streams with
  GPU-accelerated inference (TensorRT), reducing latency by 40%.
- Implemented semantic segmentation models for defect detection on a
  manufacturing line using PyTorch and OpenCV.
- Containerized models with Docker and deployed via FastAPI + Kubernetes.

Computer Vision Engineer, ImageWorks (2021 - 2022)
- Built a data labelling and augmentation pipeline for image classification
  datasets, cutting annotation time by 30%.
- Optimized ONNX Runtime inference for edge deployment on Jetson devices.

Education:
BSc in Computer Science, Hanoi University of Science and Technology (2017-2021)

Skills: Python, PyTorch, TensorFlow, OpenCV, YOLO, TensorRT, ONNX, Docker,
Kubernetes, FastAPI, Object Detection, Image Segmentation, Data Augmentation

Certifications: NVIDIA Deep Learning Institute - Computer Vision
""",
    "cv_junior_decent": """
Tran Thi Bich
Email: bich.tran@example.com | Phone: +84 902 222 333

Summary: Junior AI Engineer with 1.5 years of experience in computer
vision, passionate about deploying production ML systems.

Work Experience:
AI Engineer, SmartRetail Solutions (2023 - Present)
- Developed image classification models for product recognition using
  TensorFlow and Keras.
- Implemented object tracking (DeepSORT) for a people-counting system in
  retail stores.
- Cleaned, labelled, and augmented image datasets using Roboflow.
- Deployed models as REST APIs using Flask and Docker.

Education:
BSc in Information Technology, Da Nang University (2018-2022)

Skills: Python, TensorFlow, Keras, OpenCV, Object Detection, Image
Classification, Docker, Flask, Data Labelling
""",
    "cv_junior_weak_exp": """
Le Van Cuong
Email: cuong.le@example.com | Phone: +84 903 333 444

Summary: Recent graduate interested in AI and computer vision, completed
a 4-month internship applying deep learning to image tasks.

Work Experience:
AI Intern, University Research Lab (2024, 4 months)
- Assisted in training a CNN for handwritten digit recognition (MNIST)
  as part of a coursework project.
- Wrote data preprocessing scripts in Python and OpenCV.

Education:
BSc in Computer Science, Can Tho University (2020-2024)

Skills: Python, OpenCV, TensorFlow (coursework), Basic Machine Learning
""",
    "cv_no_degree": """
Pham Minh Duc
Email: duc.pham@example.com | Phone: +84 904 444 555

Summary: Self-taught computer vision developer with 2 years of freelance
experience building image-processing tools.

Work Experience:
Freelance Computer Vision Developer (2022 - Present)
- Built a custom OCR pipeline for invoice digitization using Tesseract and
  OpenCV for several small-business clients.
- Developed a face-detection attendance system using OpenCV and dlib.
- Deployed lightweight inference services with Flask.

Education:
High school diploma; completed online courses (Coursera Deep Learning
Specialization, fastai Practical Deep Learning).

Skills: Python, OpenCV, Tesseract OCR, Flask, dlib, Image Processing
""",
    "nlp_chatbot_strong": """
Vo Thi Hoa
Email: hoa.vo@example.com | Phone: +84 905 555 666

Summary: NLP Engineer with 3 years building chatbot and conversational
AI systems, including retrieval-augmented generation pipelines.

Work Experience:
NLP Engineer, ChatFlow AI (2022 - Present)
- Built a hybrid OCR + chatbot pipeline that extracts text from scanned
  documents and answers user questions via a RAG system using OpenAI
  embeddings and a vector database.
- Fine-tuned multi-model (text + image) pipelines combining a VLM for
  document understanding with an LLM for dialogue.
- Designed prompt templates and evaluation harnesses for chatbot quality.
- Deployed services on FastAPI with Redis caching.

Machine Learning Engineer, DataSpring (2021 - 2022)
- Built text classification models for customer-support ticket routing.

Education:
BSc in Computer Science, Vietnam National University, Hanoi (2017-2021)

Skills: Python, LLM prompting, RAG, OCR, LangChain, FastAPI, Redis,
Vector Databases, Multi-modal Models, NLP
""",
    "nlp_generalist_ok": """
Dang Quoc Bao
Email: bao.dang@example.com | Phone: +84 906 666 777

Summary: Software engineer with 1 year of experience who transitioned
into building simple chatbot features.

Work Experience:
Software Engineer, WebSoft JSC (2023 - Present)
- Integrated an OpenAI-based FAQ chatbot into the company website.
- Built basic OCR document upload feature using pytesseract.
- Maintained backend APIs in Django.

Education:
BSc in Software Engineering, FPT University (2019-2023)

Skills: Python, Django, OpenAI API, pytesseract, REST APIs
""",
    "backend_strong": """
Hoang Thi Lan
Email: lan.hoang@example.com | Phone: +84 907 777 888

Summary: Backend Engineer with 5 years building scalable Python services,
some exposure to ML model serving.

Work Experience:
Senior Backend Engineer, FinTech Vietnam (2021 - Present)
- Designed and maintained microservices in FastAPI serving 2M+ requests/day.
- Built an internal model-serving platform that wraps PyTorch models
  behind versioned REST APIs.
- Led migration from monolith to async SQLAlchemy + Postgres.

Backend Engineer, ShopHub (2019 - 2021)
- Built payment and order-processing services in Django.

Education:
BSc in Computer Science, Ho Chi Minh City University of Technology (2015-2019)

Skills: Python, FastAPI, Django, SQLAlchemy, PostgreSQL, Docker,
Kubernetes, Model Serving, System Design
""",
    "backend_junior": """
Bui Van Khoi
Email: khoi.bui@example.com | Phone: +84 908 888 999

Summary: Backend developer with 1 year of experience in REST API
development.

Work Experience:
Backend Developer, LocalTech (2023 - Present)
- Built and maintained REST APIs in FastAPI for an internal inventory
  system.
- Wrote SQL queries and basic Postgres schema migrations.

Education:
Associate Degree in Information Technology, Cao Thang Technical College
(2021-2023)

Skills: Python, FastAPI, PostgreSQL, Git, REST APIs
""",
    "data_engineer_strong": """
Do Thi Mai
Email: mai.do@example.com | Phone: +84 909 999 000

Summary: Data Engineer with 4 years building ETL pipelines and dataset
tooling for ML teams, including image/video datasets.

Work Experience:
Data Engineer, DataForge Analytics (2021 - Present)
- Built ETL pipelines processing terabytes of image and video data for
  computer vision training datasets, using Spark and Airflow.
- Implemented dataset versioning and quality checks (labelling
  consistency, deduplication, augmentation pipelines).
- Optimized GPU-based batch preprocessing jobs, cutting pipeline runtime
  by 35%.

Data Engineer, RetailData Co (2020 - 2021)
- Maintained data warehouse pipelines in Postgres and BigQuery.

Education:
BSc in Data Science, Hanoi University of Science and Technology (2016-2020)

Skills: Python, Spark, Airflow, Postgres, BigQuery, Data Augmentation,
Dataset Versioning, GPU Batch Processing
""",
    "devops_mlops": """
Ngo Van Phuc
Email: phuc.ngo@example.com | Phone: +84 910 000 111

Summary: DevOps/MLOps Engineer with 3 years deploying AI models into
production infrastructure.

Work Experience:
MLOps Engineer, CloudAI Vietnam (2022 - Present)
- Built CI/CD pipelines to containerize and deploy Computer Vision and
  NLP models via Docker and Kubernetes.
- Set up GPU-enabled Kubernetes clusters for model inference, including
  TensorRT and ONNX Runtime optimization.
- Implemented monitoring and autoscaling for model-serving APIs.

DevOps Engineer, ServerStack (2021 - 2022)
- Managed Terraform-provisioned AWS infrastructure for web services.

Education:
BSc in Information Technology, Posts and Telecommunications Institute
of Technology (2017-2021)

Skills: Docker, Kubernetes, Terraform, AWS, GPU Infrastructure, TensorRT,
ONNX Runtime, CI/CD, Monitoring
""",
    "frontend_only": """
Truong Thi Nga
Email: nga.truong@example.com | Phone: +84 911 111 222

Summary: Frontend developer with 2 years building React dashboards, no
AI/ML background.

Work Experience:
Frontend Developer, WebCraft Studio (2022 - Present)
- Built React + TypeScript dashboards for internal analytics tools.
- Integrated REST APIs and implemented responsive UI components.

Education:
BSc in Information Systems, Industrial University of Ho Chi Minh City
(2018-2022)

Skills: React, TypeScript, JavaScript, CSS, REST APIs
""",
    "qa_automation": """
Ly Van Son
Email: son.ly@example.com | Phone: +84 912 222 333

Summary: QA Automation Engineer with 2 years testing web and API
services, limited ML exposure.

Work Experience:
QA Automation Engineer, TestPro Vietnam (2022 - Present)
- Wrote automated test suites (Pytest, Selenium) for REST APIs and web UIs.
- Built a basic test harness for validating ML model output schemas.

Education:
BSc in Information Technology, Hue University (2018-2022)

Skills: Python, Pytest, Selenium, API Testing, CI/CD
""",
    "cv_phd_overqualified": """
Trinh Thi Yen
Email: yen.trinh@example.com | Phone: +84 913 333 444

Summary: PhD researcher with 8 years of academic and industry experience
in computer vision and deep learning.

Work Experience:
Research Scientist, National AI Institute (2019 - Present)
- Published research on video understanding and multi-object tracking.
- Led a team building large-scale video analytics systems for smart-city
  deployments, including object detection, tracking, and segmentation.
- Mentored junior engineers on production model deployment (Docker,
  Kubernetes, GPU optimization).

Machine Learning Engineer, VisionCorp (2016 - 2019)
- Built object detection and image classification pipelines for
  industrial inspection systems.

Education:
PhD in Computer Vision, National University of Singapore (2012-2016)
MSc in Computer Science, Vietnam National University (2010-2012)

Skills: Python, PyTorch, TensorFlow, OpenCV, Object Detection,
Segmentation, Video Analytics, GPU Optimization, Docker, Kubernetes,
Research, Deep Learning
""",
    "mobile_dev": """
Vu Thi Thanh
Email: thanh.vu@example.com | Phone: +84 914 444 555

Summary: Mobile developer with 3 years building Android/iOS apps, no
AI/CV background.

Work Experience:
Mobile Developer, AppWorks (2021 - Present)
- Built Android apps in Kotlin integrating camera and barcode-scanning
  features (using third-party SDKs, not custom ML models).
- Shipped iOS apps in Swift for e-commerce clients.

Education:
BSc in Software Engineering, Ton Duc Thang University (2017-2021)

Skills: Kotlin, Swift, Android SDK, iOS SDK, REST APIs
""",
    "ml_generalist_mid": """
Nguyen Thi Kim
Email: kim.nguyen@example.com | Phone: +84 915 555 666

Summary: Machine Learning Engineer with 2.5 years across NLP, tabular
ML, and some computer vision work.

Work Experience:
ML Engineer, DataMinds (2022 - Present)
- Built classification and recommendation models on tabular data using
  scikit-learn and XGBoost.
- Contributed to an image-tagging feature using a pretrained ResNet
  backbone with fine-tuning on a small labelled dataset.
- Built a simple chatbot FAQ feature using OpenAI's API.

Education:
BSc in Mathematics and Computer Science, Vietnam National University,
Ho Chi Minh City (2018-2022)

Skills: Python, scikit-learn, XGBoost, PyTorch (fine-tuning), OpenAI API,
Pandas, Data Analysis
""",
    "data_scientist_no_prod": """
Pham Thi Thu
Email: thu.pham@example.com | Phone: +84 916 666 777

Summary: Data Scientist with 3 years of experience in analytics and
research prototypes, limited production deployment experience.

Work Experience:
Data Scientist, MarketInsight (2021 - Present)
- Built exploratory models (regression, clustering) for market research
  reports.
- Prototyped an image-classification proof-of-concept in Jupyter
  notebooks but did not deploy it to production.

Education:
MSc in Statistics, Vietnam National University, Hanoi (2019-2021)
BSc in Mathematics, Hanoi University (2015-2019)

Skills: Python, Pandas, scikit-learn, Jupyter, Statistics, Data
Visualization
""",
}

JOBS: list[dict] = [
    {
        "title": "Senior Computer Vision Engineer",
        "jd_text": (
            "We are looking for a Senior Computer Vision Engineer to lead development of "
            "production object detection, segmentation, and video analytics systems, and "
            "to optimize models for GPU inference at scale."
        ),
        "min_years_experience": 3.0,
        "required_education_level": "bachelor",
        "responsibilities_text": (
            "Design and train object detection and segmentation models.\n"
            "Optimize inference performance on GPU (TensorRT, ONNX).\n"
            "Deploy models into production via Docker/Kubernetes.\n"
            "Mentor junior engineers on CV best practices."
        ),
    },
    {
        "title": "NLP / Chatbot Engineer",
        "jd_text": (
            "Hybrid OCR + chatbot + multi-model AI engineer role. Build retrieval-augmented "
            "chatbot systems that combine OCR document understanding with LLM-based dialogue."
        ),
        "min_years_experience": 2.0,
        "required_education_level": "bachelor",
        "responsibilities_text": (
            "Build OCR pipelines for document ingestion.\n"
            "Design RAG-based chatbot systems using LLMs.\n"
            "Integrate multi-modal (text + image) models.\n"
            "Evaluate and improve chatbot response quality."
        ),
    },
    {
        "title": "Backend Engineer (Python)",
        "jd_text": (
            "We need a Backend Engineer to build and scale FastAPI/Django services, including "
            "APIs that serve machine learning models to production traffic."
        ),
        "min_years_experience": 2.0,
        "required_education_level": "bachelor",
        "responsibilities_text": (
            "Design and maintain REST APIs in FastAPI/Django.\n"
            "Build model-serving infrastructure for ML teams.\n"
            "Own database schema design and migrations in Postgres.\n"
            "Ensure API reliability and performance at scale."
        ),
    },
    {
        "title": "Data Engineer - ML Datasets",
        "jd_text": (
            "We are hiring a Data Engineer to build ETL pipelines and dataset tooling "
            "supporting computer vision and NLP model training, including large-scale "
            "image/video data processing."
        ),
        "min_years_experience": 2.0,
        "required_education_level": "bachelor",
        "responsibilities_text": (
            "Build ETL pipelines for image/video/text datasets.\n"
            "Implement dataset versioning, deduplication, and quality checks.\n"
            "Optimize batch preprocessing jobs for GPU pipelines.\n"
            "Collaborate with ML engineers on data requirements."
        ),
    },
    {
        "title": "MLOps / DevOps Engineer",
        "jd_text": (
            "We need an MLOps Engineer to build CI/CD and infrastructure for deploying "
            "Computer Vision and NLP models into GPU-enabled production environments."
        ),
        "min_years_experience": 2.0,
        "required_education_level": "bachelor",
        "responsibilities_text": (
            "Build CI/CD pipelines for containerized ML model deployment.\n"
            "Manage GPU-enabled Kubernetes clusters for inference.\n"
            "Set up monitoring, autoscaling, and alerting for model-serving APIs.\n"
            "Optimize inference cost and latency."
        ),
    },
]


async def _get_or_create_jobs() -> list[str]:
    """Ensure the seed job postings exist; return the full list of job_ids to use for labeling."""
    async with async_session_factory() as session:
        job_ids: list[str] = []
        for job_spec in JOBS:
            job = Job(
                title=job_spec["title"],
                jd_text=job_spec["jd_text"],
                requirements_json={
                    "min_years_experience": job_spec["min_years_experience"],
                    "required_education_level": job_spec["required_education_level"],
                    "responsibilities_text": job_spec["responsibilities_text"],
                },
            )
            session.add(job)
            await session.flush()
            job_ids.append(str(job.id))
        await session.commit()
    return job_ids


async def _seed_candidates() -> None:
    """Run every resume in RESUMES through the real extraction pipeline and persist the result."""
    extraction_service = ExtractionService(build_llm_adapter())

    async with async_session_factory() as session:
        for label, resume_text in RESUMES.items():
            candidate = Candidate(raw_file_path=f"synthetic:{label}", extraction_status="pending")
            session.add(candidate)
            await session.flush()

            profile, attempts = await extraction_service.extract(resume_text)
            candidate.extraction_attempts = attempts
            if profile is None:
                candidate.extraction_status = "needs_review"
                logger.warning("Extraction failed for %s (candidate_id=%s)", label, candidate.id)
            else:
                candidate.full_name = profile.full_name
                candidate.email = profile.email
                candidate.phone = profile.phone
                candidate.structured_json = profile.model_dump(mode="json")
                candidate.extraction_status = "ok"
                logger.info("Extracted %s -> candidate_id=%s (%s)", label, candidate.id, profile.full_name)
            await session.commit()


async def main() -> None:
    """Seed jobs and candidates, then print the IDs needed to build labeled_set.jsonl."""
    job_ids = await _get_or_create_jobs()
    logger.info("Created/verified %d jobs: %s", len(job_ids), job_ids)

    await _seed_candidates()

    async with async_session_factory() as session:
        from sqlalchemy import select

        candidates = (await session.execute(select(Candidate))).scalars().all()
        jobs = (await session.execute(select(Job))).scalars().all()

    print("=== JOBS ===")
    for job in jobs:
        print(job.id, "|", job.title)

    print("=== CANDIDATES (extraction_status=ok) ===")
    for candidate in candidates:
        if candidate.extraction_status == "ok":
            print(candidate.id, "|", candidate.full_name, "|", candidate.raw_file_path)

    print("=== CANDIDATES (other status) ===")
    for candidate in candidates:
        if candidate.extraction_status != "ok":
            print(candidate.id, "|", candidate.extraction_status, "|", candidate.raw_file_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    asyncio.run(main())
