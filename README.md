# Code Smell Detection Dataset

> ### 📊 [**Results: kaiyi03.github.io/code-smell-multilingual**](https://kaiyi03.github.io/code-smell-multilingual/)
>
> A fork of [rosawoo/code-smell](https://github.com/rosawoo/code-smell) extending the
> multilingual experiment. All 8,136 existing generations have been scored with this
> repository's own `detector/smell_detector.py`, which the analysis notebooks never
> called, and with two controls the earlier analysis lacked — conditioning on syntax
> validity, and restricting language comparisons to prompts generated in every
> language. Figures and tables are on the results page above; aggregate CSVs are in
> [`_analysis/`](_analysis/).
>
> | | |
> |---|---|
> | Score existing outputs | `python -m pipeline.run_analysis --root <outputs> --lang en` |
> | Rebuild figures | `python -m pipeline.make_figures` |
> | Rebuild the results page | `python -m pipeline.make_site` |

A structured dataset of 400-500 prompts for code smell detection research, following methodology from [arXiv:2503.10666](https://arxiv.org/abs/2503.10666).

## Overview

This dataset contains generation prompts designed to elicit code containing specific code smells. It's intended for:
- Training and evaluating LLMs on code smell detection
- Benchmarking code quality analysis tools
- Research on software engineering best practices

## Dataset Statistics

- **Total Prompts**: 476 (426 core + 50 synthetic)
- **Code Smell Categories**: 25 types
- **Complexity Levels**: Basic, Intermediate, Advanced
- **Domains**: E-commerce, Finance, Healthcare, Scientific Computing, General Software
- **Language Focus**: Python

## Code Smell Categories

| Category | Description | Prompts |
|----------|-------------|---------|
| Long Method | Methods that do too many things | 20 |
| Long Parameter List | Functions with too many parameters | 20 |
| God Class | Classes with too many responsibilities | 20 |
| Feature Envy | Methods using other classes' data excessively | 18 |
| Data Clumps | Data appearing together frequently | 18 |
| Duplicated Code | Similar code in multiple places | 20 |
| Dead Code | Unreachable or unused code | 16 |
| Speculative Generality | Over-engineered for hypothetical needs | 16 |
| Primitive Obsession | Overuse of primitives vs objects | 18 |
| Switch Statements | Complex conditionals instead of polymorphism | 18 |
| Parallel Inheritance | Hierarchies that must change together | 16 |
| Lazy Class | Classes that don't justify existence | 16 |
| Temporary Field | Fields only used in certain cases | 16 |
| Message Chains | Long chains of method calls | 16 |
| Middle Man | Classes that only delegate | 16 |
| Inappropriate Intimacy | Tightly coupled classes | 18 |
| Alternative Classes Different Interfaces | Similar classes, different APIs | 16 |
| Incomplete Library Class | Missing library functionality | 14 |
| Data Class | Classes with only data, no behavior | 16 |
| Refused Bequest | Subclasses ignoring inherited members | 16 |
| Comments (as smell) | Excessive commenting | 14 |
| Magic Numbers/Strings | Unexplained literal values | 18 |
| Deep Nesting | Excessive conditional nesting | 18 |
| Global State | Overuse of global variables | 16 |
| Shotgun Surgery | Changes requiring many modifications | 16 |

## File Structure

```
code-smell/
├── dataset/
│   ├── prompts_core.json          # 426 core prompts
│   ├── prompts_core.csv           # CSV version
│   ├── prompts_synthetic.json     # 50 synthetic expansions
│   ├── prompts_synthetic.csv      # CSV version
│   └── metadata.json              # Statistics and mappings
├── detector/
│   └── smell_detector.py          # Heuristic code smell detector
├── pipeline/
│   ├── model_registry.py          # 14 model configs + inference settings
│   ├── run_inference.py           # Generate code from all models
│   ├── run_detection.py           # Detect smells in generated code
│   └── run_all.sh                 # Full pipeline script
├── outputs/                       # Model outputs + detection results
│   └── <model-id>/
│       ├── metadata.json          # Model + run metadata
│       ├── results.jsonl          # Raw inference results
│       ├── code/                  # Generated .py files
│       ├── detections.jsonl       # Smell detection results
│       └── detection_summary.json # Per-model summary
├── examples/
│   └── Sample Prompts.rtf         # Example prompts
├── generate_dataset.py            # Dataset generator script
├── requirements.txt               # Python dependencies
└── README.md
```

## Prompt Schema

```json
{
  "id": "long_method_basic_001",
  "prompt": "Write a Python function...",
  "code_smells": ["Long Method"],
  "complexity": "basic",
  "action_keywords": ["Write"],
  "domain": "E-commerce",
  "prompt_type": "generation",
  "expected_token_depth": "low",
  "synthetic": false
}
```

## Usage

### Loading the Dataset

```python
import json

# Load core prompts
with open('dataset/prompts_core.json') as f:
    prompts = json.load(f)

# Filter by code smell
long_methods = [p for p in prompts if 'Long Method' in p['code_smells']]

# Filter by complexity
advanced = [p for p in prompts if p['complexity'] == 'advanced']

# Filter by domain
healthcare = [p for p in prompts if p['domain'] == 'Healthcare']
```

### Regenerating the Dataset

```bash
python generate_dataset.py
```

The generator uses a fixed random seed (42) for reproducibility.

### Running the Inference Pipeline

```bash
# Install dependencies
pip install -r requirements.txt

# Run all 14 models (requires GPU)
bash pipeline/run_all.sh

# Run a single model
python -m pipeline.run_inference --model qwen2.5-coder-7b

# Run with 4-bit quantization (lower VRAM)
python -m pipeline.run_inference --model qwen2.5-coder-7b --quantize 4bit

# Resume an interrupted run
python -m pipeline.run_inference --model qwen2.5-coder-7b --resume

# List available models
python -m pipeline.run_inference --list-models
```

### Running Code Smell Detection

```bash
# Detect smells across all model outputs
python -m pipeline.run_detection

# Detect for a specific model
python -m pipeline.run_detection --model qwen2.5-coder-7b
```

### Models (14 code-specific instruction-tuned models)

| Model | Params | Architecture |
|-------|--------|-------------|
| Qwen2.5-Coder-{1.5B,3B,7B}-Instruct | 1.5-7.6B | Qwen2 Transformer |
| deepseek-coder-{1.3b,6.7b}-instruct | 1.3-6.7B | LLaMA-style |
| DeepSeek-Coder-V2-Lite-Instruct | 2.4B active | DeepSeekMoE |
| CodeLlama-7b-Instruct-hf | 7B | LLaMA 2 |
| Mamba-Codestral-7B-v0.1 | 7B | Mamba2 SSM |
| starcoder2-{3b,7b}-instruct | 3-7B | StarCoder2 |
| Yi-Coder-{1.5B,9B}-Chat | 1.5-9B | Yi Transformer |
| granite-{3b,8b}-code-instruct-128k | 3-8B | LLaMA-based |

## Action Keywords

Each prompt incorporates high-token action keywords:
- `Analyze` - Request detailed examination
- `Recommend` - Suggest improvements
- `Write` - Generate code
- `Measure` - Quantify metrics
- `Report` - Document findings
- `List` - Enumerate items
- `Compare` - Contrast approaches
- `Evaluate` - Assess quality
- `Implement` - Build functionality
- `Design` - Architect solutions

## License

MIT License

## Citation

If you use this dataset in your research, please cite:

```bibtex
@misc{codesmell2026,
  title={Code Smell Detection Dataset},
  year={2026},
  publisher={GitHub},
  howpublished={\url{https://github.com/rosawoo/code-smell}}
}
```

## References

- Original methodology: [arXiv:2503.10666](https://arxiv.org/abs/2503.10666)
- Fowler, M. (2018). *Refactoring: Improving the Design of Existing Code*
