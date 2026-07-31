import logging
import subprocess
import tempfile
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from config.settings import settings
from models.candidate import CandidateProfile
from utils.latex_escape import escape_latex

logger = logging.getLogger(__name__)

DEFAULT_TEMPLATE_PATH = Path("templates/company_resume_template.tex")


class TemplateService:
    """Renders a CandidateProfile into the company's LaTeX resume template and compiles it to PDF.

    Requires the `tectonic` LaTeX engine on PATH (or at `tectonic_path`):
    a self-contained binary that fetches its own package bundle, so it
    needs no separate TeX Live install.
    """

    def __init__(
        self,
        template_path: Path = DEFAULT_TEMPLATE_PATH,
        tectonic_path: str = "tectonic",
        company_name: str | None = None,
    ):
        self._template_path = template_path
        self._tectonic_path = tectonic_path
        self._company_name = company_name if company_name is not None else settings.company_name
        self._env = Environment(
            loader=FileSystemLoader(str(template_path.parent)),
            block_start_string=r"\BLOCK{",
            block_end_string="}",
            variable_start_string=r"\VAR{",
            variable_end_string="}",
            comment_start_string=r"\#{",
            comment_end_string="}",
            trim_blocks=True,
            lstrip_blocks=True,
            autoescape=False,
        )
        self._env.filters["latex"] = escape_latex

    def render(self, candidate: CandidateProfile, output_path: Path) -> Path:
        """Render `candidate` into the LaTeX template, compile to PDF, and save to `output_path`."""
        tex_source = self._render_tex(candidate)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as work_dir:
            tex_path = Path(work_dir) / "resume.tex"
            tex_path.write_text(tex_source, encoding="utf-8")

            logger.info("Compiling resume template for candidate %s", candidate.candidate_id)
            try:
                subprocess.run(
                    [self._tectonic_path, tex_path.name, "--outdir", work_dir],
                    cwd=work_dir,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                logger.exception("LaTeX compilation failed for candidate %s", candidate.candidate_id)
                raise

            compiled_pdf = Path(work_dir) / "resume.pdf"
            output_path.write_bytes(compiled_pdf.read_bytes())

        return output_path

    def _render_tex(self, candidate: CandidateProfile) -> str:
        template = self._env.get_template(self._template_path.name)
        return template.render(
            full_name=candidate.full_name,
            email=candidate.email,
            phone=candidate.phone,
            skills=candidate.skills,
            work_history=[job.model_dump() for job in candidate.work_history],
            education=[entry.model_dump() for entry in candidate.education],
            certifications=candidate.certifications,
            company_name=self._company_name,
        )
