import os
import google.generativeai as genai
import argparse
from pathlib import Path
import json
import shutil

class ModularDSAgent:
    def __init__(self, api_key, model_name="gemini-pro"):
        """Initialize the DS-Agent with Gemini API."""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.project_structure = {
            "data": ["raw", "processed"],
            "src": ["models", "utils", "config"],
            "notebooks": [],
            "tests": [],
            "docs": []
        }

    def create_project_structure(self, project_path):
        """Create a standard project structure."""
        project_path = Path(project_path)
        
        # Create main directories
        for dir_name in self.project_structure:
            (project_path / dir_name).mkdir(parents=True, exist_ok=True)
            # Create subdirectories
            for subdir in self.project_structure[dir_name]:
                (project_path / dir_name / subdir).mkdir(parents=True, exist_ok=True)

        # Create essential files
        self._create_requirements_txt(project_path)
        self._create_readme(project_path)
        self._create_gitignore(project_path)

    def _create_requirements_txt(self, project_path):
        """Create requirements.txt with necessary dependencies."""
        requirements = [
            "google-generativeai",
            "pandas",
            "numpy",
            "scikit-learn",
            "matplotlib",
            "seaborn",
            "jupyter"
        ]
        with open(project_path / "requirements.txt", "w") as f:
            f.write("\n".join(requirements))

    def _create_readme(self, project_path):
        """Create a README.md file."""
        readme_content = """# Data Science Project

This project was generated using Modular DS-Agent.

## Project Structure
- `data/`: Contains raw and processed data
- `src/`: Source code
  - `models/`: ML model implementations
  - `utils/`: Utility functions
  - `config/`: Configuration files
- `notebooks/`: Jupyter notebooks
- `tests/`: Test files
- `docs/`: Documentation

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Place your data in the `data/raw` directory
3. Run the preprocessing script:
   ```bash
   python src/utils/preprocess.py
   ```
4. Train the model:
   ```bash
   python src/models/train.py
   ```
"""
        with open(project_path / "README.md", "w") as f:
            f.write(readme_content)

    def _create_gitignore(self, project_path):
        """Create .gitignore file."""
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints

# Data
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# Environment
.env
.venv
venv/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo
"""
        with open(project_path / ".gitignore", "w") as f:
            f.write(gitignore_content)

    def analyze_problem(self, problem_description):
        """Analyze the problem and determine the type of solution needed."""
        prompt = f"""Given this data science problem:
{problem_description}

Analyze and provide a structured response in JSON format with the following:
1. problem_type: ["classification", "regression", "clustering", "time_series", "nlp"]
2. data_type: ["tabular", "text", "image", "time_series"]
3. required_libraries: list of Python libraries needed
4. suggested_approach: brief description of the approach
5. evaluation_metrics: list of appropriate metrics
"""
        response = self.model.generate_content(prompt)
        return json.loads(response.text)

    def generate_solution(self, problem_description, analysis, project_path):
        """Generate solution scripts based on problem analysis."""
        project_path = Path(project_path)
        
        # Generate preprocessing script
        self._generate_preprocessing_script(analysis, project_path)
        
        # Generate model script
        self._generate_model_script(analysis, project_path)
        
        # Generate training script
        self._generate_training_script(analysis, project_path)
        
        # Generate evaluation script
        self._generate_evaluation_script(analysis, project_path)

    def _generate_preprocessing_script(self, analysis, project_path):
        """Generate data preprocessing script."""
        prompt = f"""Create a Python script for data preprocessing based on this analysis:
{json.dumps(analysis, indent=2)}

The script should:
1. Load data from data/raw
2. Handle missing values
3. Perform feature engineering
4. Save processed data to data/processed
"""
        response = self.model.generate_content(prompt)
        
        script_path = project_path / "src" / "utils" / "preprocess.py"
        with open(script_path, "w") as f:
            f.write(response.text)

    def _generate_model_script(self, analysis, project_path):
        """Generate model implementation script."""
        prompt = f"""Create a Python script for the ML model based on this analysis:
{json.dumps(analysis, indent=2)}

The script should:
1. Define the model architecture
2. Include necessary imports
3. Include model configuration
"""
        response = self.model.generate_content(prompt)
        
        script_path = project_path / "src" / "models" / "model.py"
        with open(script_path, "w") as f:
            f.write(response.text)

    def _generate_training_script(self, analysis, project_path):
        """Generate model training script."""
        prompt = f"""Create a Python script for model training based on this analysis:
{json.dumps(analysis, indent=2)}

The script should:
1. Load preprocessed data
2. Split data into train/validation sets
3. Train the model
4. Save the trained model
"""
        response = self.model.generate_content(prompt)
        
        script_path = project_path / "src" / "models" / "train.py"
        with open(script_path, "w") as f:
            f.write(response.text)

    def _generate_evaluation_script(self, analysis, project_path):
        """Generate model evaluation script."""
        prompt = f"""Create a Python script for model evaluation based on this analysis:
{json.dumps(analysis, indent=2)}

The script should:
1. Load the trained model
2. Load test data
3. Make predictions
4. Calculate evaluation metrics
5. Generate visualizations
"""
        response = self.model.generate_content(prompt)
        
        script_path = project_path / "src" / "models" / "evaluate.py"
        with open(script_path, "w") as f:
            f.write(response.text)

def main():
    parser = argparse.ArgumentParser(description="Modular DS-Agent for single problem execution")
    parser.add_argument("--api-key", required=True, help="Gemini API key")
    parser.add_argument("--problem", required=True, help="Problem description file or text")
    parser.add_argument("--output-dir", required=True, help="Output directory for the project")
    args = parser.parse_args()

    # Initialize DS-Agent
    agent = ModularDSAgent(args.api_key)

    # Read problem description
    if os.path.isfile(args.problem):
        with open(args.problem, 'r') as f:
            problem_description = f.read()
    else:
        problem_description = args.problem

    # Create project structure
    agent.create_project_structure(args.output_dir)

    # Analyze problem
    analysis = agent.analyze_problem(problem_description)
    print("Problem Analysis:", json.dumps(analysis, indent=2))

    # Generate solution
    agent.generate_solution(problem_description, analysis, args.output_dir)
    print(f"Project generated successfully in {args.output_dir}")

if __name__ == "__main__":
    main() 