import os
import re
import json
import tempfile
from typing import List, Dict, Tuple, Optional, Set
import streamlit as st
from pdfminer.high_level import extract_text as extract_text_from_pdf_lib
from dotenv import load_dotenv
from dataclasses import dataclass, asdict
from openai import OpenAI
import hashlib
import time
import traceback
from pathlib import Path
import pandas as pd # Import pandas for cleaner table display

# Load environment variables
load_dotenv()

# Configuration
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# --- API KEY CONFIGURATION ---
API_KEY = os.getenv('OPENAI_API_KEY')
if not API_KEY:
    st.warning("⚠️ **OPENAI_API_KEY environment variable not set!** Please set your API key to enable advanced formula extraction from documents.")
    MOCK_MODE = True
else:
    client = OpenAI(api_key=API_KEY)
    MOCK_MODE = False

# --- INPUT VARIABLES DEFINITIONS ---
INPUT_VARIABLES = {
    'TERM_START_DATE': 'Date when the policy starts',
    'FUP_Date': 'First Unpaid Premium date',
    'ENTRY_AGE': 'Age of the policyholder at policy inception',
    'TOTAL_PREMIUM': 'Annual Premium amount',
    'BOOKING_FREQUENCY': 'Frequency of premium booking (monthly, quarterly, yearly)',
    'PREMIUM_TERM': 'Premium Paying Term - duration for paying premiums',
    'SUM_ASSURED': 'Sum Assured - guaranteed amount on maturity/death',
    'Income_Benefit_Amount': 'Amount of income benefit',
    'Income_Benefit_Frequency': 'Frequency of income benefit payout',
    'DATE_OF_SURRENDER': 'Date when policy is surrendered',
    'no_of_premium_paid': 'Years passed since date of commencement till FUP',
    'maturity_date': 'Date of commencement + (BENEFIT_TERM * 12 months)',
    'policy_year': 'Years passed + 1 between date of commencement and surrender date',
    'BENEFIT_TERM': 'The duration (in years) for which the policy benefits are payable',
    'GSV_FACTOR': 'Guaranteed Surrender Value Factor, a percentage used to calculate the minimum guaranteed surrender value from total premiums paid.',
    'SSV1_FACTOR': 'Surrender Value Factor',
    'SSV3_FACTOR': 'A special factor used to compute Special Surrender Value (SSV) related to paid-up income benefits',
    'SSV2_FACTOR':'A special factor used to compute Special Surrender Value (SSV) related to return of premium (ROP)',
    'FUND_VALUE': 'The total value of the policy fund at the time of surrender or maturity',
    'SYSTEM_PAID': 'The amount paid by the system for surrender or maturity'
}

# Basic derived formulas that can be logically computed
BASIC_DERIVED_FORMULAS = {
    'no_of_premium_paid': 'Calculate based on difference between TERM_START_DATE and FUP_Date',
    'policy_year': 'Calculate based on difference between TERM_START_DATE and DATE_OF_SURRENDER + 1',
    'maturity_date': 'TERM_START_DATE + (BENEFIT_TERM* 12) months',
    'Final_surrender_value_paid':'Final surrender value paid',
    'CAPITAL_FUND_VALUE': 'The total value of the policy fund at the time of surrender or maturity, including any bonuses or additional benefits',
    'FUND_FACTOR': 'A factor used to compute the fund value based on the total premiums paid and the policy term'
}

# Default TARGET OUTPUT VARIABLES (formulas must be extracted from document)
DEFAULT_TARGET_OUTPUT_VARIABLES = [
    'TOTAL_PREMIUM_PAID',
    'TEN_TIMES_AP',
    'one_oh_five_percent_total_premium',
    'SUM_ASSURED_ON_DEATH',
    'GSV',
    'PAID_UP_SA',
    'PAID_UP_SA_ON_DEATH',
    'paid_up_income_benefit_amount',
    'SSV1_AMT',
    'SSV2_AMT',
    'SSV3_AMT',
    'SSV',
    'SURRENDER_PAID_AMOUNT',
    'PV',
    'N',
    'SURRENDER_CHARGE_VALUE',
    'SURRENDER_CHARGE',
    'FINAL_SURRENDER_VALUE',
]

@dataclass
class ExtractedFormula:
    formula_name: str
    formula_expression: str
    variants_info: str
    business_context: str
    confidence: float
    source_method: str
    document_evidence: str
    specific_variables: Dict[str, str]

    def to_dict(self):
        return asdict(self)

@dataclass
class DocumentExtractionResult:
    input_variables: Dict[str, str]
    basic_derived_formulas: Dict[str, str]
    extracted_formulas: List[ExtractedFormula]
    extraction_summary: str
    overall_confidence: float

    def to_dict(self):
        return asdict(self)

class DocumentFormulaExtractor:
    """Extracts formulas purely from document content without hardcoded formulas"""

    def __init__(self, target_outputs: List[str]):
        self.input_variables = INPUT_VARIABLES
        self.basic_derived = BASIC_DERIVED_FORMULAS
        self.target_outputs = target_outputs # Now dynamic

    def extract_formulas_from_document(self, text: str) -> DocumentExtractionResult:
        """Extract all formulas from document text"""

        if MOCK_MODE or not API_KEY:
            return self._explain_no_extraction()

        try:
            st.info("🔍 Starting document-based formula extraction...")

            # Progress bar
            progress_bar = st.progress(0)

            # First, analyze document structure and identify formula sections
            progress_bar.progress(10)
            formula_sections = self._identify_formula_sections(text)

            # Extract surrender value formula with special focus
            progress_bar.progress(20)
            surrender_result = self._extract_surrender_formula_specifically(text)

            # Extract all other formulas from document
            extracted_formulas = []
            target_outputs_for_loop = [f.upper() for f in self.target_outputs]

            total_formulas = len(target_outputs_for_loop)

            for i, formula_name in enumerate(target_outputs_for_loop):
                # Ensure surrender_value is not extracted twice if already handled
                if formula_name.lower() == 'surrender_value' and surrender_result:
                    continue

                progress_bar.progress(20 + int((i / total_formulas) * 70))

                formula_result = self._extract_formula_from_document(text, formula_name, formula_sections)
                if formula_result:
                    extracted_formulas.append(formula_result)

                time.sleep(0.1)  # Rate limiting

            # Add the specifically extracted surrender result if it exists and hasn't been added
            if surrender_result and not any(f.formula_name == 'SURRENDER_VALUE' for f in extracted_formulas):
                extracted_formulas.insert(0, surrender_result) # Add it to the beginning for prominence
            
            progress_bar.progress(100)

            # Calculate overall confidence
            overall_conf = sum(f.confidence for f in extracted_formulas) / len(extracted_formulas) if extracted_formulas else 0.0

            st.success("✅ Formula extraction complete!")

            return DocumentExtractionResult(
                input_variables=self.input_variables,
                basic_derived_formulas=self.basic_derived,
                extracted_formulas=extracted_formulas,
                extraction_summary=f"Document analysis complete. Successfully identified {len(extracted_formulas)} formulas from your document.",
                overall_confidence=overall_conf,
            )
            
        except Exception as e:
            st.error(f"❌ **Document extraction failed:** {e}")
            st.exception(e) # Display full traceback for debugging
            return self._explain_no_extraction()
    
    def _identify_formula_sections(self, text: str) -> List[str]:
        """Identify sections of document that contain formulas"""
        
        prompt = f"""
        Analyze this insurance document and identify all sections that contain mathematical formulas, calculations, or surrender value computations.
        
        DOCUMENT: {text}
        
        TASK: Extract ONLY the text sections that contain:
        1. Mathematical formulas (with = signs, calculations)
        2. Surrender value calculations (GSV, SSV, SSV1, SSV2, SSV3)
        3. Benefit calculations
        4. Premium calculations
        5. Any text that shows how values are computed
        
        Return each relevant section as a separate block.
        Focus especially on surrender value, GSV, SSV, paid-up calculations.
        
        FORMAT: Return sections separated by "---SECTION---"
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            response_text = response.choices[0].message.content
            sections = response_text.split("---SECTION---")
            return [section.strip() for section in sections if section.strip()]
            
        except Exception as e:
            st.error(f"Error identifying formula sections: {e}")
            return [text]
        
    def _extract_formula_from_document(self, text: str, formula_name: str, formula_sections: List[str]) -> Optional[ExtractedFormula]:
        """Extract specific formula from document sections with variable identification"""
        
        search_text = "\n".join(formula_sections) if formula_sections else text
        
        ssv_context = ""
        # Convert formula_name to lower for robust comparison
        if formula_name.lower() in ['ssv1_amt', 'ssv2_amt', 'ssv3_amt']:
            ssv_context = """
            NOTE: For SSV (Special Surrender Value) components:
            - SSV1 typically relates to present value of paid-up sum assured on death
            - SSV2 typically relates to ROP (Return of Premium) or Total Premiums paid benefit
            - SSV3 typically relates to paid-up income instalments or survival benefits
            Look for these specific components in the surrender value calculations.
            """
        
        prompt = f"""
        Extract the formula for "{formula_name}" from this insurance document content.
        
        DOCUMENT CONTENT: {search_text}
        
        AVAILABLE VARIABLES: {list(self.input_variables.keys())}
        BASIC DERIVED: {self.basic_derived}
        
        TARGET: Find how "{formula_name}" is calculated in this document.
        {ssv_context}
        
        INSTRUCTIONS:
        1. Look for explicit formulas, calculation methods, or mathematical relationships
        2. Express using only the available variable names above. Prioritize exact variable names, if not found, use a descriptive placeholder.
        3. IDENTIFY ONLY THE SPECIFIC VARIABLES used in this formula
        4. Extract exact supporting text from document
        5. If not explicitly stated but can be logically derived from context, derive it
        6. If truly not found or derivable, return "NOT_FOUND"
        
        RESPONSE FORMAT:
        FORMULA: [mathematical expression using only relevant variables]
        SPECIFIC_VARIABLES: [comma-separated list of variables actually used in this formula]
        DOCUMENT_EVIDENCE: [exact text that supports this]
        CONTEXT: [business explanation]
        METHOD: [EXPLICIT/DERIVED/NOT_FOUND]
        CONFIDENCE: [0.1-1.0]
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            response_text = response.choices[0].message.content
            
            if "NOT_FOUND" in response_text:
                return None
                
            return self._parse_formula_response(response_text, formula_name)
            
        except Exception as e:
            st.error(f"Error extracting {formula_name}: {e}")
            return None
    
    def _extract_surrender_formula_specifically(self, text: str) -> Optional[ExtractedFormula]:
        """Special focused extraction for surrender value formula"""
        
        prompt = f"""
        CRITICAL TASK: Extract the surrender value calculation formula from this insurance document.
        
        DOCUMENT: {text}
        
        AVAILABLE INPUT VARIABLES: {list(self.input_variables.keys())}
        BASIC DERIVED FORMULAS: {self.basic_derived}
        
        REQUIREMENTS:
        1. Find the EXACT surrender value calculation method from the document
        2. Express formula using only the available variable names above. Prioritize exact variable names, if not found, use a descriptive placeholder.
        3. IDENTIFY ONLY THE SPECIFIC VARIABLES used in surrender value calculation
        4. If document mentions GSV (Guaranteed Surrender Value) and SSV (Special Surrender Value), show relationship
        5. If multiple variants exist, show all variants
        6. Extract the ACTUAL text from document that describes this calculation.
        7. ROP= Total premiums paid. Display total premiums in the formula wherever ROP is used.
        
        RESPONSE FORMAT:
        SURRENDER_FORMULA: [exact mathematical expression using available variables]
        SPECIFIC_VARIABLES: [comma-separated list of variables actually used]
        VARIANTS: [if multiple calculation methods exist]
        DOCUMENT_EVIDENCE: [exact text from document that supports this formula]
        BUSINESS_LOGIC: [explanation of when/how this applies]
        CONFIDENCE: [0.1-1.0 based on clarity in document]
        
        If surrender value is not clearly defined in document, respond with "NOT_FOUND"
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            response_text = response.choices[0].message.content
            
            if "NOT_FOUND" in response_text:
                return None
                
            return self._parse_surrender_response(response_text)
            
        except Exception as e:
            st.error(f"Error extracting surrender formula: {e}")
            return None
    
    def _parse_surrender_response(self, response_text: str) -> Optional[ExtractedFormula]:
        """Parse surrender formula response"""
        
        try:
            formula_match = re.search(r'SURRENDER_FORMULA:\s*(.+?)(?=\nSPECIFIC_VARIABLES|$)', response_text, re.DOTALL | re.IGNORECASE)
            formula_expression = formula_match.group(1).strip() if formula_match else "Formula not clearly defined"
            
            variables_match = re.search(r'SPECIFIC_VARIABLES:\s*(.+?)(?=\nVARIANTS|$)', response_text, re.DOTALL | re.IGNORECASE)
            specific_vars_str = variables_match.group(1).strip() if variables_match else ""
            specific_variables = self._parse_specific_variables(specific_vars_str)
            
            variants_match = re.search(r'VARIANTS:\s*(.+?)(?=\nDOCUMENT_EVIDENCE|$)', response_text, re.DOTALL | re.IGNORECASE)
            variants_info = variants_match.group(1).strip() if variants_match else "Single method"
            
            evidence_match = re.search(r'DOCUMENT_EVIDENCE:\s*(.+?)(?=\nBUSINESS_LOGIC|$)', response_text, re.DOTALL | re.IGNORECASE)
            document_evidence = evidence_match.group(1).strip() if evidence_match else "Evidence not extracted"
            
            logic_match = re.search(r'BUSINESS_LOGIC:\s*(.+?)(?=\nCONFIDENCE|$)', response_text, re.DOTALL | re.IGNORECASE)
            business_context = logic_match.group(1).strip() if logic_match else "Surrender value calculation"
            
            confidence_match = re.search(r'CONFIDENCE:\s*([0-9]*\.?[0-9]+)', response_text, re.IGNORECASE)
            confidence = float(confidence_match.group(1)) if confidence_match else 0.5
            
            return ExtractedFormula(
                formula_name='SURRENDER_VALUE',
                formula_expression=formula_expression,
                variants_info=variants_info,
                business_context=business_context,
                confidence=confidence,
                source_method='document_extraction',
                document_evidence=document_evidence,
                specific_variables=specific_variables
            )
            
        except Exception as e:
            st.error(f"Error parsing surrender response: {e}")
            return None
    
    def _parse_formula_response(self, response_text: str, formula_name: str) -> Optional[ExtractedFormula]:
        """Parse general formula response"""
        
        try:
            formula_match = re.search(r'FORMULA:\s*(.+?)(?=\nSPECIFIC_VARIABLES|$)', response_text, re.DOTALL | re.IGNORECASE)
            formula_expression = formula_match.group(1).strip() if formula_match else "Formula not found"
            
            variables_match = re.search(r'SPECIFIC_VARIABLES:\s*(.+?)(?=\nDOCUMENT_EVIDENCE|$)', response_text, re.DOTALL | re.IGNORECASE)
            specific_vars_str = variables_match.group(1).strip() if variables_match else ""
            specific_variables = self._parse_specific_variables(specific_vars_str)
            
            evidence_match = re.search(r'DOCUMENT_EVIDENCE:\s*(.+?)(?=\nCONTEXT|$)', response_text, re.DOTALL | re.IGNORECASE)
            document_evidence = evidence_match.group(1).strip() if evidence_match else "No supporting text found"
            
            context_match = re.search(r'CONTEXT:\s*(.+?)(?=\nMETHOD|$)', response_text, re.DOTALL | re.IGNORECASE)
            business_context = context_match.group(1).strip() if context_match else f"Calculation for {formula_name}"
            
            method_match = re.search(r'METHOD:\s*(.+?)(?=\nCONFIDENCE|$)', response_text, re.IGNORECASE)
            method = method_match.group(1).strip() if method_match else "UNKNOWN"
            
            confidence_match = re.search(r'CONFIDENCE:\s*([0-9]*\.?[0-9]+)', response_text, re.IGNORECASE)
            confidence = float(confidence_match.group(1)) if confidence_match else 0.3
            
            return ExtractedFormula(
                formula_name=formula_name.upper(),
                formula_expression=formula_expression,
                variants_info=f"Extraction method: {method}",
                business_context=business_context,
                confidence=confidence,
                source_method='document_extraction',
                document_evidence=document_evidence,
                specific_variables=specific_variables
            )
            
        except Exception as e:
            st.error(f"Error parsing formula response for {formula_name}: {e}")
            return None
    
    def _parse_specific_variables(self, variables_str: str) -> Dict[str, str]:
        """Parse specific variables from comma-separated string"""
        
        specific_variables = {}
        if variables_str:
            var_names = [var.strip() for var in variables_str.split(',')]
            
            for var_name in var_names:
                if var_name in self.input_variables:
                    specific_variables[var_name] = self.input_variables[var_name]
                elif var_name in self.basic_derived:
                    specific_variables[var_name] = self.basic_derived[var_name]
                else:
                    specific_variables[var_name] = f"Variable used in calculation: {var_name}"
        
        return specific_variables
    
    def _explain_no_extraction(self) -> DocumentExtractionResult:
        """Explain that extraction cannot be performed without API key"""
        
        return DocumentExtractionResult(
            input_variables=self.input_variables,
            basic_derived_formulas=self.basic_derived,
            extracted_formulas=[],
            extraction_summary="Cannot extract formulas from document. A valid `OPENAI_API_KEY` is required to enable advanced document analysis and formula extraction. This system is designed to identify complex calculations, especially for surrender values, directly from your provided insurance policy documents.",
            overall_confidence=0.0,
        )

def extract_text_from_file(file_bytes, file_extension):
    """Extract text from supported file formats"""
    try:
        if file_extension == '.pdf':
            # Create a temporary file for PDF processing
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                tmp_file.write(file_bytes)
                tmp_file_path = tmp_file.name
            
            try:
                text = extract_text_from_pdf_lib(tmp_file_path)
                os.unlink(tmp_file_path)  # Clean up
                return text
            except Exception as e:
                os.unlink(tmp_file_path)  # Clean up on error
                raise e
        
        elif file_extension == '.txt':
            return file_bytes.decode('utf-8')
        
        elif file_extension == '.docx':
            try:
                import docx
                from io import BytesIO
                doc = docx.Document(BytesIO(file_bytes))
                return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
            except ImportError:
                st.error("`python-docx` not installed. Please install it: `pip install python-docx`")
                return ""
        
        else:
            return ""
            
    except Exception as e:
        st.error(f"Error extracting text from file: {e}")
        return ""



# --- UI Styling and Helper Functions ---
def set_custom_css():
    st.markdown(
        """
        <style>
       @import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@700&display=swap');
       @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600&display=swap');

        html, body, .main {
            font-family: 'Roboto', sans-serif;
            background: linear-gradient(135deg, #a6d3ff 50%, #cbdff7 100%);
            color: #2c3e50;
            animation: fadeIn 1s ease-in-out;
            overflow-x: hidden;
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        /* Professional button styling */
        .stButton > button {
            background: linear-gradient(135deg, #a6d3ff 50%, #cbdff7 100%);
            color: white !important;
            padding: 12px 25px;
            border-radius: 10px;
            border: none;
            cursor: pointer;
            font-color: #ffffff !important;
            font-size: 17px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(0, 77, 168, 0.3);
            letter-spacing: 0.5px;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg,  #004DA8 25%, #1e88e5 50%);
            transform: translateY(-3px) scale(1.02);
            box-shadow: 0 6px 18px rgba(0, 77, 168, 0.4);
            filter: brightness(1.05);
            font-color: #ffffff !important;
            color: white !important;
        }
        .stButton > button:active {
            transform: translateY(0);
            box-shadow: 0 4px 12px rgba(0, 77, 168, 0.3);
            color: white !important;
        }

        /* Professional expander styling */
        .streamlit-expander > div[role="button"] {
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%) !important;
            border-left: 6px solid #004DA8 !important;
            padding: 12px 15px;
            border-radius: 10px;
            margin-bottom: 8px;
            transition: all 0.3s ease;
            box-shadow: 0 2px 5px rgba(0, 77, 168, 0.1);
            font-weight: 500;
            color: #1a1a1a !important;
            font-family: 'Montserrat', sans-serif;
            font-size: 1.05em;
        }
        .streamlit-expander > div[role="button"]:hover {
            background: linear-gradient(135deg, #bbdefb 0%, #90caf9 100%) !important;
            transform: translateX(5px);
            box-shadow: 0 3px 8px rgba(0, 77, 168, 0.2);
            color: #1a1a1a !important;
        }
        .streamlit-expander .streamlit-expanderContent {
            background-color: #ffffff;
            border-radius: 0 0 10px 10px;
            padding: 20px;
            margin-top: -8px;
            border: 1px solid #e3f2fd;
            border-top: none;
            box-shadow: 0 2px 8px rgba(0, 77, 168, 0.05);
            color: #2c3e50 !important;
        }

        /* Professional header styling */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Montserrat', sans-serif !important;
            color: #004DA8 !important;
            font-weight: 700;
            margin-top: 1.8em;
            margin-bottom: 0.8em;
            text-shadow: 1px 1px 2px rgba(0, 77, 168, 0.1);
            animation: slideIn 0.8s ease-out;
        }
        @keyframes slideIn {
            from { transform: translateY(-10px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }

        /* Professional metric styling */
        .stMetric {
            background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 15px rgba(0, 77, 168, 0.1);
            text-align: center;
            transition: all 0.3s ease;
            margin-bottom: 15px;
            border: 1px solid #e0e0e0;
        }
        .stMetric:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 20px rgba(0, 77, 168, 0.15);
            background: linear-gradient(135deg, #f8fbff 0%, #e3f2fd 100%);
        }
        .stMetric [data-testid="metric-container"] > div:first-child {
            font-size: 1.1em;
            color: #fafcff !important;
            font-weight: 500;
        }
        .stMetric [data-testid="metric-container"] > div:nth-child(2) {
            font-size: 2.2em;
            font-weight: bold;
            color: #fafcff !important;
            margin-top: 5px;
        }

        /* Professional inline code styling */
        .stMarkdown code {
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            color: #004DA8 !important;
            border-radius: 5px;
            padding: 3px 6px;
            font-family: 'Source Code Pro', monospace;
            font-size: 0.9em;
            border: 1px solid #e0e0e0;
        }

        /* Professional dataframe styling */
        .stDataFrame table {
            font-family: 'Roboto', sans-serif;
            width: 100%;
            border-collapse: collapse;
            margin-top: 25px;
            box-shadow: 0 5px 15px rgba(0, 77, 168, 0.1);
            border-radius: 12px;
            overflow: hidden;
            background-color: white;
            border: 1px solid #e3f2fd;
        }
        .stDataFrame thead th {
            background: linear-gradient(135deg, #004DA8 0%, #1976d2 100%);
            color: white !important;
            padding: 15px 20px;
            text-align: left;
            border-bottom: 2px solid #1565c0;
            font-weight: 700;
            font-size: 1.05em;
        }
        .stDataFrame tbody td {
            padding: 12px 20px;
            border-bottom: 1px solid #e3f2fd;
            font-size: 0.95em;
            color: #2c3e50 !important;
        }
        .stDataFrame tbody tr:nth-child(even) {
            background: linear-gradient(135deg, #f8fbff 0%, #ffffff 100%);
        }
        .stDataFrame tbody tr:hover {
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        }

        /* Professional alert styling */
        .stAlert[data-baseweb="notification"] {
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }
        .stAlert[data-baseweb="notification"][kind="success"] {
            background: linear-gradient(135deg, #e8f5e8 0%, #d4edda 100%);
            color: #155724 !important;
            border-left: 5px solid #28a745;
        }
        .stAlert[data-baseweb="notification"][kind="warning"] {
            background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
            color: #856404 !important;
            border-left: 5px solid #ffc107;
        }
        .stAlert[data-baseweb="notification"][kind="info"] {
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            color: #004DA8 !important;
            border-left: 5px solid #004DA8;
        }
        .stAlert[data-baseweb="notification"][kind="error"] {
            background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
            color: #721c24 !important;
            border-left: 5px solid #dc3545;
        }

        /* Professional text input styling */
        .stTextInput label {
            font-weight: 600;
            color: #0 !important;
        }
        .stTextInput input {
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            padding: 8px 12px;
            transition: all 0.2s ease;
            color: #f7fbff !important;
            background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
        }
        .stTextInput input:focus {
            border-color: #e0e0e0;
            box-shadow: 0 0 0 0.2rem rgba(0, 77, 168, 0.25);
            outline: none;
            background: white;
        }
        .stTextInput input::placeholder {
            color: #6c757d !important;
        }

        /* Professional multiselect styling */
        .stMultiSelect label {
            font-weight: 600;
            color: #f5f7fa !important;
        }
        .stMultiSelect > div > div {
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
        }
        .stMultiSelect > div > div:focus-within {
            border-color: #e0e0e0;
            box-shadow: 0 0 0 0.2rem rgba(0, 77, 168, 0.25);
            background: white;
        }
        .stMultiSelect [data-baseweb="tag"] {
            background: linear-gradient(135deg, #6d96c7 0%, #a6d3ff 50%, #cbdff7 100%);
            color: white !important;
            border-radius: 5px;
            padding: 4px 10px;
            font-size: 0.9em;
            font-weight: 500;
        }
        .stMultiSelect [data-baseweb="tag"] svg {
            color: white !important;
        }
        .stMultiSelect input {
            color: #2f3b4a !important;
        }

        /* Main app container text color */
        .stApp > div:first-child {
            color: #f5f7fa !important;
        }
       /* Replace the problematic broad rule with more specific ones */

/* Main app container text color - only for light backgrounds */
.stApp > div:first-child {
    color: #f7fbff !important;
}

/* Specific text elements for light backgrounds only */
.stApp .stMarkdown:not(.stSidebar .stMarkdown), 
.stApp .stText:not(.stSidebar .stText), 
.stApp p:not(.stSidebar p), 
.stApp div:not(.stSidebar div):not([data-testid*="stSelectbox"]):not([data-testid*="stMultiSelect"]), 
.stApp span:not(.stSidebar span) {
    color: #2c3e50 !important;
}

/* Blue headers for professional look - only for light backgrounds */
.stMarkdown h1:not(.stSidebar h1), 
.stMarkdown h2:not(.stSidebar h2), 
.stMarkdown h3:not(.stSidebar h3), 
.stMarkdown h4:not(.stSidebar h4), 
.stMarkdown h5:not(.stSidebar h5), 
.stMarkdown h6:not(.stSidebar h6) {
    color: #004DA8 !important;
}


/* Dark blue buttons and tags - white text */
.stButton > button {
    color: white !important;
}

.stButton > button:hover {
    color: white !important;
}

.stButton > button:active {
    color: white !important;
}

/* Multi-select tags - white text */
.stMultiSelect [data-baseweb="tag"] {
    color: white !important;
}

.stMultiSelect [data-baseweb="tag"] span {
    color: white !important;
}

.stMultiSelect [data-baseweb="tag"] svg {
    color: white !important;
}

/* Any dark blue background elements */
[style*="background: linear-gradient(135deg, #004DA8"],
[style*="background-color: #004DA8"],
[style*="background: #004DA8"],
div[style*="background: linear-gradient(135deg, #004DA8"] * {
    color: white !important;
}

/* Force white text on dark blue backgrounds */
.stApp [style*="background: linear-gradient(135deg, #004DA8"] {
    color: white !important;
}

.stApp [style*="background: linear-gradient(135deg, #004DA8"] * {
    color: white !important;
}

/* File uploader dark areas */
.stFileUploader [data-testid="stFileUploaderDropzone"] {
    background: rgba(255, 255, 255, 0.1) !important;
    color: #ffffff !important;
}
    
        
        
        /* Blue headers for professional look */
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
            color: #004DA8 !important;
        }
        
        /* Professional file uploader styling */
        .stFileUploader label {
            color: black !important;
            font-weight: 600;
        }
        
        .stFileUploader > div > div {
            background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
            border: 2px dashed #90caf9;
            border-radius: 8px;
            transition: all 0.3s ease;
            color: #2f3b4a !important;
        }
        
        .stFileUploader > div > div:hover {
            border-color: #e0e0e0;
            background: linear-gradient(135deg, #f8fbff 0%, #e3f2fd 100%);
        }
        
        .stFileUploader [data-testid="stFileUploaderDropzone"] {
            color: #2f3b4a !important;
        }
        
        /* Professional selectbox styling */
        .stSelectbox label {
            color: #f7fbff !important;
            font-weight: 600;
        }
        
        .stSelectbox > div > div {
            border-radius: 8px;
            border: 1px solid #acadad;
            background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
            color: #f7fbff !important;
        }
        
        .stSelectbox > div > div:focus-within {
            border-color: #e0e0e0;
            box-shadow: 0 0 0 0.2rem rgba(0, 77, 168, 0.25);
        }
        
        .stSelectbox [data-baseweb="select"] {
            color: #f7fbff !important;
        }

    
        
        /* Professional slider styling */
        .stSlider label {
            color: #2c3e50 !important;
            font-weight: 600;
        }
        
        .stSlider [data-baseweb="slider"] {
            background: #e3f2fd !important;
        }
        
        .stSlider [data-baseweb="slider"] [data-testid="stSlider"] {
            color: #004DA8 !important;
        }
        
        /* Professional number input styling */
        .stNumberInput label {
            color: #2c3e50 !important;
            font-weight: 600;
        }
        
        .stNumberInput input {
            border-radius: 8px;
            border: 1px solid #acadad;
            background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
            color: #2c3e50 !important;
        }
        
        .stNumberInput input:focus {
            border-color: #e0e0e0;
            box-shadow: 0 0 0 0.2rem rgba(0, 77, 168, 0.25);
        }
        
        /* Professional checkbox styling */
        .stCheckbox label {
            color: #f7fbff !important;
            font-weight: 500;
        }
        
        /* Professional radio button styling */
        .stRadio label {
            color: #ffffff !important;
            font-weight: 600;
        }
        
        .stRadio [data-testid="stRadio"] > div {
            color: #2c3e50 !important;
        }
        
        /* Professional date input styling */
        .stDateInput label {
            color: #fffff !important;
            font-weight: 600;
        }
        
        .stDateInput input {
            border-radius: 8px;
            border: 1px solid #acadad;
            background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
            color: #2c3e50 !important;
        }
        
        .stDateInput input:focus {
            border-color: #e0e0e0;
            box-shadow: 0 0 0 0.2rem rgba(0, 77, 168, 0.25);
        }
        
        /* Professional time input styling */
        .stTimeInput label {
            color: #acadad !important;
            font-weight: 600;
        }
        
        .stTimeInput input {
            border-radius: 8px;
            border: 1px solid #acadad;
            background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
            color: #2c3e50 !important;
        }
        
        .stTimeInput input:focus {
            border-color: #e0e0e0;
            box-shadow: 0 0 0 0.2rem rgba(0, 77, 168, 0.25);
        }
        </style>
        """,
        unsafe_allow_html=True
    )
def main():
    st.set_page_config(
        page_title="Document Formula Extractor",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Apply custom CSS
    set_custom_css()

    st.title("Document Formula Extractor")
    st.markdown("Your tool to **automatically extract mathematical formulas** from insurance policy documents using advanced AI capabilities.")

    # Initialize session state
    if 'extraction_result' not in st.session_state:
        st.session_state.extraction_result = None
    if 'selected_output_variables' not in st.session_state:
        st.session_state.selected_output_variables = DEFAULT_TARGET_OUTPUT_VARIABLES.copy()
    if 'custom_output_variable' not in st.session_state:
        st.session_state.custom_output_variable = ""
    # Ensure custom variables list exists
    if 'user_defined_output_variables' not in st.session_state:
        st.session_state.user_defined_output_variables = []


    # --- Main Content Area ---
    st.markdown("---") # Separator for main content

    col1, col2 = st.columns([1, 1])

    with col1:
        

        # Output Variable Selection
        st.subheader(" Select Target Formulas")
        st.markdown("Choose which formulas you want to extract from the document. You can also add custom ones!")

        # Combine default and user-defined variables for the multiselect options
        all_possible_output_variables = sorted(list(set(DEFAULT_TARGET_OUTPUT_VARIABLES + st.session_state.user_defined_output_variables)))

        # Update selected_output_variables with the multiselect
        st.session_state.selected_output_variables = st.multiselect(
            "Formulas to search for:",
            options=all_possible_output_variables,
            default=st.session_state.selected_output_variables,
            help="These are the key formulas the system will try to find. Select all that apply."
        )

        # Add custom output variable
        st.session_state.custom_output_variable = st.text_input(
            "Add a custom formula name (e.g., 'DEATH_BENEFIT_CALC'):",
            value=st.session_state.custom_output_variable,
            key="custom_output_input",
            help="Enter a specific formula name you want to extract, even if it's not in the default list. Press 'Add' to include it."
        )

        if st.button(" Add Custom Formula", key="add_custom_formula_button"):
            new_var = st.session_state.custom_output_variable.strip()
            if new_var and new_var not in st.session_state.user_defined_output_variables and new_var not in DEFAULT_TARGET_OUTPUT_VARIABLES:
                st.session_state.user_defined_output_variables.append(new_var)
                # Also automatically select the newly added variable
                if new_var not in st.session_state.selected_output_variables:
                    st.session_state.selected_output_variables.append(new_var)
                st.session_state.custom_output_variable = "" # Clear input after adding
                st.success(f"'{new_var}' added and selected!")
                st.rerun() # Rerun to update the multiselect widget
            elif new_var in st.session_state.user_defined_output_variables or new_var in DEFAULT_TARGET_OUTPUT_VARIABLES:
                st.info(f"'{new_var}' is already in the list of available formulas.")
            else:
                st.warning("Please enter a valid custom formula name to add.")

        st.markdown("---") # Visual separation

        st.subheader("⬆ Upload Product Specifications")
        st.markdown("Upload your insurance policy document (PDF, TXT, DOCX) to begin formula extraction.")

        uploaded_file = st.file_uploader(
            "Select a document",
            type=list(ALLOWED_EXTENSIONS),
            help=f"Accepts: {', '.join(ALLOWED_EXTENSIONS)}. Max file size: {MAX_FILE_SIZE / (1024*1024):.1f} MB",
            key="file_uploader"
        )

        if uploaded_file is not None:
            if uploaded_file.size > MAX_FILE_SIZE:
                st.error(f"File size exceeds the limit. Please upload a file smaller than {MAX_FILE_SIZE / (1024*1024):.1f} MB.")
                st.session_state.extraction_result = None
                return

            st.info(f" **File Selected:** `{uploaded_file.name}` (`{uploaded_file.size / 1024:.1f} KB`)")

            if st.button(" Analyze Document", type="primary", key="analyze_button"):
                if not st.session_state.selected_output_variables:
                    st.warning("Please select at least one target formula to extract or add a custom one.")
                    st.session_state.extraction_result = None
                    return

                with st.spinner("Analyzing document and extracting formulas... This may take a moment."):
                    file_extension = Path(uploaded_file.name).suffix.lower()
                    text = extract_text_from_file(uploaded_file.read(), file_extension)

                    if not text.strip():
                        st.error("❗ Could not extract readable text from the uploaded file. Please ensure it contains text content.")
                        st.session_state.extraction_result = None
                        return

                    # Only proceed with extraction if API key is configured
                    if not MOCK_MODE and API_KEY:
                        extractor = DocumentFormulaExtractor(target_outputs=st.session_state.selected_output_variables)
                        extraction_result = extractor.extract_formulas_from_document(text)
                        st.session_state.extraction_result = extraction_result
                    else:
                        st.session_state.extraction_result = None # Clear result if no API key
                        st.warning("⚠️ Cannot perform extraction without a configured OPENAI_API_KEY.")
        else:
            st.session_state.extraction_result = None

    with col2:
        st.subheader(" Reference Variables")
        st.markdown("These variables provide context for the AI during formula identification. Expand sections below to view details.")

        # Using individual expanders for each category to keep it compact
        with st.expander("Input Variables (Policy Parameters)", expanded=False):
            st.markdown("These are the **fundamental policy parameters** that serve as building blocks for calculations.")
            input_data = [{"Variable": name, "Description": desc} for name, desc in INPUT_VARIABLES.items()]
            st.dataframe(pd.DataFrame(input_data), use_container_width=True, hide_index=True)

        with st.expander("Basic Derived Formulas (Pre-computed)", expanded=False):
            st.markdown("These represent **common values logically computed** from the input variables, often without explicit document formulas.")
            derived_data = [{"Variable": name, "Description": desc} for name, desc in BASIC_DERIVED_FORMULAS.items()]
            st.dataframe(pd.DataFrame(derived_data), use_container_width=True, hide_index=True)

        with st.expander("Currently Selected Target Output Variables", expanded=True):
            st.markdown("The system will actively **search for and extract formulas** corresponding to these variables from your document.")
            # Ensure only unique variables are shown in case of custom additions
            current_targets = sorted(list(set(st.session_state.selected_output_variables)))
            if current_targets:
                target_data = [{"Target Variable": var} for var in current_targets]
                st.dataframe(pd.DataFrame(target_data), use_container_width=True, hide_index=True)
            else:
                st.info("No target formulas selected yet. Use the multiselect or add custom formulas on the left.")

    st.markdown("---")

    # --- Display Results ---
    if st.session_state.extraction_result:
        st.subheader("Extraction Summary")

        result = st.session_state.extraction_result

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric(label="Total Formulas Found", value=len(result.extracted_formulas), help="Number of distinct formulas successfully extracted.")
        with col_s2:
            st.metric(label="Overall Confidence", value=f"{result.overall_confidence:.1%}", help="Average confidence score across all extracted formulas.")

        st.info(result.extraction_summary)

        if result.extracted_formulas:
            st.subheader(" Detailed Formula Overview")
            st.markdown("Review the extracted formulas, their expressions, and supporting evidence.")

            formula_data = []
            for formula in result.extracted_formulas:
                formula_data.append({
                    "Formula Name": formula.formula_name,
                    "Expression": formula.formula_expression,
                    "Confidence": f"{formula.confidence:.1%}",
                    "Business Context": formula.business_context,
                    "Document Evidence": formula.document_evidence,
                    "Specific Variables": ", ".join(formula.specific_variables.keys())
                })

            df = pd.DataFrame(formula_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.markdown("---")

            st.subheader("⬇ Export Extracted Data")
            st.markdown("Download the extracted formulas in various formats for further analysis or integration.")

            export_data = {
                "extraction_summary": result.extraction_summary,
                "total_formulas": len(result.extracted_formulas),
                "overall_confidence": result.overall_confidence,
                "formulas": [formula.to_dict() for formula in result.extracted_formulas]
            }

            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                st.download_button(
                    label=" Download as JSON",
                    data=json.dumps(export_data, indent=2),
                    file_name="extracted_formulas.json",
                    mime="application/json",
                    help="Export all detailed formula data in JSON format."
                )

            with col_exp2:
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label=" Download as CSV",
                    data=csv_data,
                    file_name="extracted_formulas.csv",
                    mime="text/csv",
                    help="Export a summarized table of formulas in CSV format."
                )
        else:
            st.warning("No formulas were extracted based on your selections and the document content. Try selecting more general terms or ensuring your API key is set.")

    st.markdown("---")

    # Footer
    st.markdown(
        """
        <div style="text-align: center; margin-top: 50px; color: #7f8c8d; font-size: 0.9em;">
            <p>Developed with ❤️ using Streamlit and OpenAI API</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()