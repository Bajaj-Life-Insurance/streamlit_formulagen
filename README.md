# Formula Generator from Natural Language Documents

This project extracts mathematical formulas from unstructured insurance policy documents using advanced prompt engineering with OpenAI's GPT models. It is tailored for use cases such as extracting surrender value, premium, GSV, SSV, and benefit calculations from policy PDFs and text files.

---

## Features

- Extracts explicit and derived formulas directly from document content
- Identifies variables used in each formula
- Parses document evidence and business logic for each computation
- Displays confidence score for each extracted formula
- Highlights key sections from documents that contain relevant calculations

---

## How It Works

1. **Document Upload**: Upload an insurance policy document in PDF, TXT, or DOCX format.
2. **Section Identification**: The model identifies and isolates sections likely to contain formulas.
3. **Formula Extraction**: For each target variable (e.g., SURRENDER_VALUE, SSV1, PREMIUM), GPT-4o is prompted to extract the corresponding formula.
4. **Evidence and Context Parsing**: Supporting text, business logic, and variable references are extracted alongside.
5. **UI Display**: Extracted formulas and metadata are displayed in a Streamlit interface.

---

## Prompting Techniques Used

This project leverages advanced prompt engineering strategies to improve the reliability and clarity of outputs:

### 1. Task-Specific Instructions
Each prompt includes a clear objective, such as:
"Extract the formula for "SSV2_AMT" from this document."
This keeps the model focused on a specific outcome.

### 2. Variable Grounding
Prompts include known input and derived variables. The model is instructed to use only these, reducing hallucinations:
AVAILABLE VARIABLES: [AGE, TERM, SUM_ASSURED, ...]

### 3. Structured Output Format
All responses follow a predefined format for consistency and easy parsing:
FORMULA:
SPECIFIC_VARIABLES:
DOCUMENT_EVIDENCE:
CONTEXT:
METHOD:
CONFIDENCE:

### 4. Domain Heuristics
Prompts include helpful hints for common insurance terms (e.g., SSV1, ROP), improving contextual understanding.

### 5. Negative Case Handling
The model is allowed to return `NOT_FOUND` when a formula can't be extracted, avoiding forced or inaccurate responses.

### 6. Confidence Scoring
Each output includes a model-estimated confidence score to help judge reliability:

### 7. Focused Section Analysis
Before formula extraction, the model identifies only the relevant parts of the document, reducing noise.

---

