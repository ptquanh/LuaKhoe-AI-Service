import os
import shutil
import re

def main():
    base_dir = r"d:\CODE PLAYGROUND\Projects\Fullstack\Project_LuaKhoe\LuaKhoe-AI"
    os.chdir(base_dir)

    # 1. Create directories
    dirs = [
        "src/modules/system",
        "src/modules/predict",
        "src/modules/analyze",
        "src/modules/advisory",
        "src/shared/middlewares",
        "src/shared/utils",
        "src/shared/constants"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    # 2. File moves map
    moves = {
        "api/main.py": "src/main.py",
        "api/routers/system.py": "src/modules/system/controller.py",
        "api/routers/predict.py": "src/modules/predict/controller.py",
        "src/core_inference.py": "src/modules/predict/service.py",
        "api/routers/analyze.py": "src/modules/analyze/controller.py",
        "api/routers/rag.py": "src/modules/advisory/controller.py",
        "src/rag/vector_store.py": "src/modules/advisory/repository.py",
        "src/rag/models.py": "src/modules/advisory/models.py",
        "src/rag/prompts.py": "src/modules/advisory/prompts.py",
        "src/rag/document_parser.py": "src/modules/advisory/document_parser.py",
        "src/rag/ingestion.py": "src/modules/advisory/ingestion.py",
        "src/rag/recommendation_graph.py": "src/modules/advisory/recommendation_graph.py",
        "src/database.py": "src/shared/database.py",
        "api/dependencies.py": "src/shared/middlewares/dependencies.py",
        "src/logger.py": "src/shared/utils/logger.py",
        "src/storage.py": "src/shared/utils/storage.py",
        "src/validation.py": "src/shared/utils/validation.py",
        "api/utils.py": "src/shared/utils/helpers.py"
    }

    for src_file, dst_file in moves.items():
        if os.path.exists(src_file):
            print(f"Moving {src_file} -> {dst_file}")
            shutil.move(src_file, dst_file)
        else:
            print(f"WARN: {src_file} not found!")

    # 3. Create DTO files
    schemas_content = ""
    if os.path.exists("api/schemas.py"):
        with open("api/schemas.py", "r", encoding="utf-8") as f:
            schemas_content = f.read()

    system_dto = """from pydantic import BaseModel
from typing import List

class APIStatus(BaseModel):
    status: str
    ai_strategy: str
    storage_strategy: str
    labels: List[str]
    rag_enabled: bool = False
    rag_chunks_count: int = 0
"""
    predict_dto = """from pydantic import BaseModel
from typing import Optional

class PredictionResult(BaseModel):
    disease: str
    confidence: float
    status: str
    model_version: str
    latency_ms: float
    saved_path: str
    filename: str
    low_confidence: Optional[bool] = False
"""
    advisory_dto = """from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class TreatmentProtocol(BaseModel):
    chemical: str = "Không có dữ liệu trong tài liệu tham khảo"
    biological: str = "Không có dữ liệu trong tài liệu tham khảo"
    cultural: str = "Không có dữ liệu trong tài liệu tham khảo"

class RecommendationResult(BaseModel):
    disease_name: str
    severity_assessment: str
    immediate_actions: List[str]
    treatment_protocol: TreatmentProtocol
    npk_adjustment: str
    prevention_measures: List[str]
    sources_used: List[str]
    confidence_note: str

class RecommendationRequest(BaseModel):
    disease: str = Field(..., description="Detected disease name from CV model")
    confidence: float = Field(..., ge=0.0, le=1.0, description="CV model confidence score")

class RecommendationResponse(BaseModel):
    status: str
    recommendation: RecommendationResult
    latency_ms: float
    rag_chunks_used: int

class IngestionRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Raw text content to ingest")
    source: str = Field(..., description="Source document name (e.g., 'IRRI_blast_2023.pdf')")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional extra metadata")

class IngestionResponse(BaseModel):
    status: str
    message: str
"""
    analyze_dto = """from pydantic import BaseModel
from typing import Optional
from src.modules.predict.dto import PredictionResult
from src.modules.advisory.dto import RecommendationResponse

class AnalyzeResponse(BaseModel):
    prediction: PredictionResult
    recommendation: Optional[RecommendationResponse] = None
"""

    with open("src/modules/system/dto.py", "w", encoding="utf-8") as f: f.write(system_dto)
    with open("src/modules/predict/dto.py", "w", encoding="utf-8") as f: f.write(predict_dto)
    with open("src/modules/advisory/dto.py", "w", encoding="utf-8") as f: f.write(advisory_dto)
    with open("src/modules/analyze/dto.py", "w", encoding="utf-8") as f: f.write(analyze_dto)

    if os.path.exists("api/schemas.py"):
        os.remove("api/schemas.py")

    # 4. Global string replacements for imports
    replacements = {
        "api.routers.system": "src.modules.system.controller",
        "api.routers.predict": "src.modules.predict.controller",
        "api.routers.analyze": "src.modules.analyze.controller",
        "api.routers.rag": "src.modules.advisory.controller",
        "api.dependencies": "src.shared.middlewares.dependencies",
        "src.core_inference": "src.modules.predict.service",
        "src.rag.vector_store": "src.modules.advisory.repository",
        "src.rag.models": "src.modules.advisory.models",
        "src.rag.prompts": "src.modules.advisory.prompts",
        "src.rag.document_parser": "src.modules.advisory.document_parser",
        "src.rag.ingestion": "src.modules.advisory.ingestion",
        "src.rag.recommendation_graph": "src.modules.advisory.recommendation_graph",
        "src.database": "src.shared.database",
        "src.logger": "src.shared.utils.logger",
        "src.storage": "src.shared.utils.storage",
        "src.validation": "src.shared.utils.validation",
        "api.utils": "src.shared.utils.helpers",
    }

    # Special logic for api.schemas
    # Since schemas are split, we need to replace specific imports
    # from api.schemas import APIStatus -> from src.modules.system.dto import APIStatus
    # and so on. We can do regex for this.
    
    schema_map = {
        "APIStatus": "src.modules.system.dto",
        "PredictionResult": "src.modules.predict.dto",
        "TreatmentProtocol": "src.modules.advisory.dto",
        "RecommendationResult": "src.modules.advisory.dto",
        "RecommendationRequest": "src.modules.advisory.dto",
        "RecommendationResponse": "src.modules.advisory.dto",
        "IngestionRequest": "src.modules.advisory.dto",
        "IngestionResponse": "src.modules.advisory.dto",
        "AnalyzeResponse": "src.modules.analyze.dto"
    }

    def process_file(filepath):
        if not os.path.isfile(filepath):
            return
        if filepath.endswith(".py"):
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            
            orig_content = content
            for old, new in replacements.items():
                content = content.replace(old, new)
            
            # Fix schemas imports
            # Match: from api.schemas import A, B, C
            def schema_replacer(match):
                imports_str = match.group(1)
                imports = [i.strip() for i in imports_str.split(",")]
                res = []
                for imp in imports:
                    if imp in schema_map:
                        res.append(f"from {schema_map[imp]} import {imp}")
                    else:
                        res.append(f"from api.schemas import {imp}") # fallback
                return "\n".join(res)
            
            content = re.sub(r'from\s+api\.schemas\s+import\s+([A-Za-z0-9_,\s]+)', schema_replacer, content)

            # Some files might do `import api.schemas as schemas` or similar, we'll need to check manually later.
            if content != orig_content:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)

    for root, dirs, files in os.walk("src"):
        for file in files:
            process_file(os.path.join(root, file))
    
    for root, dirs, files in os.walk("tests"):
        if "tests" in dirs: # just check
            pass
        for file in files:
            process_file(os.path.join(root, file))
            
    print("Refactoring done.")

if __name__ == "__main__":
    main()
