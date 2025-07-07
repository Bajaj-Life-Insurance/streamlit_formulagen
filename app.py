import os
import re
import json
import tempfile
from typing import List, Dict, Tuple, Optional, Set
import streamlit as st
from pdfminer.high_level import extract_text as extract_text_from_pdf_lib
import google.generativeai as genai
from dotenv import load_dotenv
from dataclasses import dataclass, asdict
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
API_KEY = os.getenv('GEMINI_API_KEY')
if not API_KEY:
    st.warning("⚠️ GEMINI_API_KEY environment variable not set! Please set your API key to extract formulas from documents.")
    MOCK_MODE = True
else:
    genai.configure(api_key=API_KEY)
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
    'SSV2_FACTOR':'A special factor used to compute Special Surrender Value (SSV) related to return of premium (ROP)'
}

# Basic derived formulas that can be logically computed
BASIC_DERIVED_FORMULAS = {
    'no_of_premium_paid': 'Calculate based on difference between TERM_START_DATE and FUP_Date',
    'policy_year': 'Calculate based on difference between TERM_START_DATE and DATE_OF_SURRENDER + 1',
    'maturity_date': 'TERM_START_DATE + (BENEFIT_TERM* 12) months',
    'Final_surrender_value_paid':'Final surrender value paid'
}

# TARGET OUTPUT VARIABLES (formulas must be extracted from document)
TARGET_OUTPUT_VARIABLES = [
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
    'PV'
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
    
    def __init__(self):
        self.input_variables = INPUT_VARIABLES
        self.basic_derived = BASIC_DERIVED_FORMULAS
        self.target_outputs = TARGET_OUTPUT_VARIABLES
        
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
            total_formulas = len(self.target_outputs)
            
            for i, formula_name in enumerate(self.target_outputs):
                # Ensure surrender_value is not extracted twice if already handled
                if formula_name.lower() == 'surrender_value' and surrender_result:
                    continue 

                progress_bar.progress(20 + int((i / total_formulas) * 70))
                
                formula_result = self._extract_formula_from_document(text, formula_name, formula_sections)
                if formula_result:
                    extracted_formulas.append(formula_result)
                
                time.sleep(0.2)  # Rate limiting
            
            # Add the specifically extracted surrender result if it exists and hasn't been added
            if surrender_result and not any(f.formula_name == 'surrender_value' for f in extracted_formulas):
                extracted_formulas.insert(0, surrender_result) # Add it to the beginning for prominence
            
            progress_bar.progress(100)
            surrender_found = any(f.formula_name == 'surrender_value' for f in extracted_formulas)
            
            return DocumentExtractionResult(
                input_variables=self.input_variables,
                basic_derived_formulas=self.basic_derived,
                extracted_formulas=extracted_formulas,
                extraction_summary=f"Document analysis complete. Extracted {len(extracted_formulas)} formulas from source document.",
                overall_confidence=sum(f.confidence for f in extracted_formulas) / len(extracted_formulas) if extracted_formulas else 0.0,
            )
            
        except Exception as e:
            st.error(f"❌ Document extraction failed: {e}")
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
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            
            sections = response.text.split("---SECTION---")
            return [section.strip() for section in sections if section.strip()]
            
        except Exception as e:
            st.error(f"Error identifying formula sections: {e}")
            return [text]
    
    def _extract_formula_from_document(self, text: str, formula_name: str, formula_sections: List[str]) -> Optional[ExtractedFormula]:
        """Extract specific formula from document sections with variable identification"""
        
        search_text = "\n".join(formula_sections) if formula_sections else text
        
        ssv_context = ""
        if formula_name.lower() in ['ssv1_amt', 'ssv2_amt', 'ssv3_amt']: # Changed to match TARGET_OUTPUT_VARIABLES case
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
        2. Express using only the available variable names above
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
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            
            if "NOT_FOUND" in response.text:
                return None
                
            return self._parse_formula_response(response.text, formula_name)
            
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
        2. Express formula using only the available variable names above
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
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            
            if "NOT_FOUND" in response.text:
                return None
                
            return self._parse_surrender_response(response.text)
            
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
                formula_name='SURRENDER_VALUE', # Using consistent casing for display
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
                formula_name=formula_name.upper(), # Ensure consistent casing
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
            extraction_summary="Cannot extract formulas from document. A valid `GEMINI_API_KEY` is required to enable advanced document analysis and formula extraction. This system is designed to identify complex calculations, especially for surrender values, directly from your provided insurance policy documents.",
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
        .main {
            background: linear-gradient(to right, #e0f2f7, #ffffff); /* Light blue to white gradient */
            color: #333333; /* Dark grey for text */
        }
        .stButton button {
            background-color: #4CAF50; /* Green */
            color: white;
            padding: 10px 20px;
            border-radius: 8px;
            border: none;
            cursor: pointer;
            font-size: 16px;
            transition: all 0.3s ease;
        }
        .stButton button:hover {
            background-color: #45a049;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        .stExpander div[role="button"] {
            background-color: #f0f0f0; /* Light grey for expanders */
            border-left: 5px solid #a7d9f7; /* Light blue border */
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 5px;
            transition: all 0.2s ease;
        }
        .stExpander div[role="button"]:hover {
            background-color: #e0e0e0;
        }
        .stExpander .streamlit-expanderContent {
            background-color: #f8f8f8; /* Slightly lighter for content */
            border-radius: 8px;
            padding: 15px;
            margin-top: -10px; /* Adjust to prevent gap */
        }
        h1, h2, h3, h4, h5, h6 {
            color: #2c3e50; /* Dark blue/grey for headers */
            font-weight: 600;
        }
        .stMetric {
            background-color: #f0f8ff; /* AliceBlue for metrics */
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            text-align: center;
        }
        .stMetric > div > div:first-child {
            font-size: 0.9em;
            color: #6c757d; /* Muted grey for label */
        }
        .stMetric > div > div:nth-child(2) > div:first-child {
            font-size: 1.8em;
            font-weight: bold;
            color: #007bff; /* Bright blue for value */
        }
        .stMarkdown code {
            background-color: #e6f7ff; /* Light blue for code blocks */
            color: #0056b3; /* Darker blue for code text */
            border-radius: 4px;
            padding: 2px 4px;
        }
        /* Custom Styling for the Table (using Streamlit's default table for simplicity) */
        .dataframe {
            font-family: Arial, sans-serif;
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            border-radius: 8px;
            overflow: hidden; /* Ensures border-radius is applied */
        }
        .dataframe th {
            background-color: #cfe2f3; /* Lighter blue header */
            color: #333333;
            padding: 12px 15px;
            text-align: left;
            border-bottom: 2px solid #a7d9f7; /* Matching border */
            font-weight: 600;
        }
        .dataframe td {
            padding: 10px 15px;
            border-bottom: 1px solid #dee2e6; /* Light grey border */
        }
        .dataframe tr:nth-child(even) {
            background-color: #f7f7f7; /* Very light grey for even rows */
        }
        .dataframe tr:hover {
            background-color: #e9ecef; /* Slightly darker grey on hover */
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
    
    st.title(" Document Formula Extractor")
    st.markdown("Your tool to **automatically extract mathematical formulas** from insurance policy documents using advanced AI capabilities.")
    
    # Initialize session state
    if 'extraction_result' not in st.session_state:
        st.session_state.extraction_result = None
    
    
    # --- Main Content Area ---
    st.markdown("---") # Separator for main content

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("⬆ Upload Product Specifications")
        st.markdown("Upload your insurance policy document (PDF, TXT, DOCX) to begin formula extraction.")
        
        uploaded_file = st.file_uploader(
            "Select a document",
            type=list(ALLOWED_EXTENSIONS),
            help=f"Accepts: {', '.join(ALLOWED_EXTENSIONS)}. Max file size: {MAX_FILE_SIZE / (1024*1024):.1f} MB",
            key="file_uploader" # Added a key for stability
        )
        
        if uploaded_file is not None:
            if uploaded_file.size > MAX_FILE_SIZE:
                st.error(f"File size exceeds the limit. Please upload a file smaller than {MAX_FILE_SIZE / (1024*1024):.1f} MB.")
                st.session_state.extraction_result = None # Clear previous results if file is too large
                return
            
            st.info(f" **File Selected:** `{uploaded_file.name}` (`{uploaded_file.size / 1024:.1f} KB`)")
            
            # Use a clearer button for action
            if st.button(" Analyze Document", type="primary"):
                with st.spinner("Analyzing document and extracting formulas..."):
                    file_extension = Path(uploaded_file.name).suffix.lower()
                    text = extract_text_from_file(uploaded_file.read(), file_extension)
                    
                    if not text.strip():
                        st.error("❗ Could not extract readable text from the uploaded file. Please ensure it contains text content.")
                        st.session_state.extraction_result = None
                        return
                    
                    st.success(f"✔️ Successfully extracted {len(text):,} characters for analysis.")
                    
                    extractor = DocumentFormulaExtractor()
                    extraction_result = extractor.extract_formulas_from_document(text)
                    
                    st.session_state.extraction_result = extraction_result
        else:
            # Clear results if no file is uploaded or cleared
            st.session_state.extraction_result = None
    
    with col2:
        st.subheader(" Reference Variables")
        st.markdown("Understand the context for formula identification.")
        
        # Use a single expander with tabs for better organization and sleekness
        with st.expander("Explore Defined Variables & Formulas", expanded=True):
            tab1, tab2, tab3 = st.tabs(["Input Variables", "Basic Derived", "Target Outputs"])
            
            with tab1:
                st.markdown("These are the fundamental policy parameters.")
                for var_name, description in INPUT_VARIABLES.items():
                    st.markdown(f"- **{var_name}:** `{description}`")
            
            with tab2:
                st.markdown("Formulas that can be logically computed from input variables.")
                for var_name, description in BASIC_DERIVED_FORMULAS.items():
                    st.markdown(f"- **{var_name}:** `{description}`")
            
            with tab3:
                st.markdown("The key formulas to be extracted from the document.")
                for var_name in TARGET_OUTPUT_VARIABLES:
                    st.markdown(f"- **{var_name}**")

    st.markdown("---")

    # --- Display Results ---
    if st.session_state.extraction_result:
        st.subheader("✨ Extraction Summary")
        
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

            # Prepare data for a clean DataFrame
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
            
            # Create a DataFrame and display as a table
            df = pd.DataFrame(formula_data)
            st.dataframe(df, use_container_width=True, hide_index=True) # use_container_width for responsiveness, hide_index for cleaner look

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
                csv_data = df.to_csv(index=False) # Use pandas to_csv for direct CSV export
                st.download_button(
                    label=" Download as CSV",
                    data=csv_data,
                    file_name="extracted_formulas.csv",
                    mime="text/csv",
                    help="Export a summarized table of formulas in CSV format."
                )
        else:
            st.warning("No formulas were extracted from the document. Please check the document content or your API key configuration.")
    
    st.markdown("---")

    
if __name__ == "__main__":
    main()