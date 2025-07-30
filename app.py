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
from openai import AzureOpenAI
import hashlib
import time
import traceback
from pathlib import Path
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import numpy as np
from collections import defaultdict


load_dotenv()


# Configuration
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB


AZURE_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
DEPLOYMENT_NAME = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
AZURE_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION')

if not AZURE_API_KEY:
    st.warning("⚠️ **AZURE_OPENAI_API_KEY environment variable not set!** Please set your API key to enable advanced formula extraction from documents.")
    MOCK_MODE = True
    client = None
else:
    try:
        client = AzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            api_version=AZURE_API_VERSION
        )
        MOCK_MODE = False

    except Exception as e:
        st.error(f"❌ Failed to initialize Azure OpenAI client: {str(e)}")
        MOCK_MODE = True
        client = None


# --- STABLE CHUNKING CONFIGURATION ---
STABLE_CHUNK_CONFIG = {
    'chunk_size': 1500,  # Stable chunk sizede
    'chunk_overlap': 300,  # 20% overlap ratio
    'max_chunks_per_formula': 5,  # Limit chunks processed per formula
    'relevance_threshold': 0.3,  # Minimum relevance score
    'min_chunk_size': 500,  # Minimum viable chunk size
    'max_context_length': 4000,  # Maximum context for API calls
}

# --- QUOTA MANAGEMENT CONFIGURATION ---
QUOTA_CONFIG = {
    'max_retries': 3,
    'retry_delay': 5,  # seconds
    'fallback_enabled': True,
    'batch_size': 5,  # Process formulas in batches
    'emergency_stop_after_failures': 10,  # Stop after this many consecutive failures
}

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
    'N':'min(Policy_term, 20) - Elapsed_policy_duration',
    'SYSTEM_PAID': 'The amount paid by the system for surrender or maturity',
    'CAPITAL_UNITS_VALUE':' The number of units in the policy fund at the time of surrender or maturity',

}

# Basic derived formulas that can be logically computed
BASIC_DERIVED_FORMULAS = {
    'no_of_premium_paid': 'Calculate based on difference between TERM_START_DATE and FUP_Date',
    'policy_year': 'Calculate based on difference between TERM_START_DATE and DATE_OF_SURRENDER + 1',
    'maturity_date': 'TERM_START_DATE + (BENEFIT_TERM* 12) months',
    'Final_surrender_value_paid':'Final surrender value paid',
    'Elapsed_policy_duration': 'How many years have passed since policy start',
    'CAPITAL_FUND_VALUE': 'The total value of the policy fund at the time of surrender or maturity, including any bonuses or additional benefits',
    'FUND_FACTOR': 'A factor used to compute the fund value based on the total premiums paid and the policy term'
}

# Default TARGET OUTPUT VARIABLES (formulas must be extracted from document)
DEFAULT_TARGET_OUTPUT_VARIABLES = [
    'TOTAL_PREMIUM_PAID',
    'TEN_TIMES_AP',
    'one_oh_five_percent_total_premium',
    'SUM_ASSURED_ON_DEATH',
    'SUM_ASSURED',
    'GSV',
    'PAID_UP_SA',
    'PAID_UP_SA_ON_DEATH',
    'paid_up_income_benefit_amount',
    'SSV1_AMT',
    'SSV2_AMT',
    'SSV3_AMT',
    'SSV',
    'SURRENDER_PAID_AMOUNT',
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

class StableChunkedDocumentFormulaExtractor:
    """Extracts formulas from large documents using stable chunking ratios"""

    def __init__(self, target_outputs: List[str]):
        self.input_variables = INPUT_VARIABLES
        self.basic_derived = BASIC_DERIVED_FORMULAS
        self.target_outputs = target_outputs
        self.config = STABLE_CHUNK_CONFIG
        self.quota_config = QUOTA_CONFIG
        self.failure_count = 0  # Track consecutive failures
        
        # Initialize text splitter with stable configuration
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config['chunk_size'],
            chunk_overlap=self.config['chunk_overlap'],
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""]
        )
        
        # Formula keywords for relevance scoring
        self.formula_keywords = {
            'high_priority': ['surrender', 'gsv', 'ssv', 'formula', 'calculate', 'premium', 'benefit'],
            'medium_priority': ['paid-up', 'maturity', 'death', 'sum assured', 'charge', 'value'],
            'low_priority': ['policy', 'term', 'amount', 'date', 'factor', 'rate']
        }

    def _extract_formula_offline(self, formula_name: str, context: str) -> Optional[ExtractedFormula]:
        """Offline formula extraction using pattern matching as fallback"""
        
        # Common formula patterns to look for
        formula_patterns = {
            'equals': r'([A-Z_]+)\s*[=:]\s*([^.\n]+)',
            'calculation': r'([A-Z_]+)\s*(?:is calculated|calculated as|=|:)\s*([^.\n]+)',
            'formula': r'(?:formula|calculation)\s*(?:for|of)\s*([A-Z_]+)[:\s]*([^.\n]+)',
            'definition': r'([A-Z_]+)\s*(?:means|refers to|defined as)\s*([^.\n]+)'
        }
        
        # Search for the specific formula name
        formula_lower = formula_name.lower()
        context_lower = context.lower()
        
        # Look for direct mentions
        if formula_lower in context_lower:
            # Find the sentence containing the formula
            sentences = context.split('.')
            relevant_sentences = []
            
            for sentence in sentences:
                if formula_lower in sentence.lower():
                    relevant_sentences.append(sentence.strip())
            
            if relevant_sentences:
                # Try to extract formula from relevant sentences
                for sentence in relevant_sentences:
                    for pattern_name, pattern in formula_patterns.items():
                        matches = re.findall(pattern, sentence, re.IGNORECASE)
                        if matches:
                            # Found a potential formula
                            formula_expr = matches[0][1] if len(matches[0]) > 1 else matches[0][0]
                            
                            # Extract variables mentioned in the formula
                            variables_found = []
                            for var in self.input_variables.keys():
                                if var.lower() in formula_expr.lower():
                                    variables_found.append(var)
                            
                            return ExtractedFormula(
                                formula_name=formula_name.upper(),
                                formula_expression=formula_expr.strip(),
                                variants_info="Extracted using offline pattern matching",
                                business_context=f"Offline extraction for {formula_name}",
                                confidence=0.3,  # Lower confidence for offline extraction
                                source_method='offline_pattern_matching',
                                document_evidence=sentence[:200] + "..." if len(sentence) > 200 else sentence,
                                specific_variables={var: self.input_variables[var] for var in variables_found}
                            )
        
        # If no direct match found, return None
        return None

    def extract_formulas_from_document(self, text: str) -> DocumentExtractionResult:
        """Extract formulas from large document using stable chunking strategy"""
        
        if MOCK_MODE or not AZURE_API_KEY:
            return self._explain_no_extraction()

        try:
            progress_bar = st.progress(0)

            # Step 1: Create stable chunks
            progress_bar.progress(10)
            chunks = self._create_stable_chunks(text)
    

            # Step 2: Score and rank chunks
            progress_bar.progress(20)
            scored_chunks = self._score_chunks_for_relevance(chunks)

            # Step 3: Check API status before starting
            if not self._check_api_status():
                st.error("❌ API is not accessible. Using offline extraction only.")
                return self._fallback_to_offline_extraction(text)

            # Step 4: Extract formulas using quota-aware batching
            progress_bar.progress(30)
            extracted_formulas = []
            total_formulas = len(self.target_outputs)
            
            # Process formulas in batches to manage quota
            batch_size = self.quota_config['batch_size']
            for batch_start in range(0, len(self.target_outputs), batch_size):
                batch_end = min(batch_start + batch_size, len(self.target_outputs))
                batch_formulas = self.target_outputs[batch_start:batch_end]
                
                
                for i, formula_name in enumerate(batch_formulas):
                    overall_progress = 30 + int(((batch_start + i) / total_formulas) * 60)
                    progress_bar.progress(overall_progress)
                    
                    # Check if we should stop due to too many failures
                    if self.failure_count >= self.quota_config['emergency_stop_after_failures']:
                        st.error(f"🛑 Emergency stop: {self.failure_count} consecutive failures. Switching to offline mode.")
                        remaining_formulas = self.target_outputs[batch_start + i:]
                        for remaining_formula in remaining_formulas:
                            offline_result = self._extract_formula_offline(remaining_formula, text)
                            if offline_result:
                                extracted_formulas.append(offline_result)
                        break
                    
                    # Use stable extraction for each formula
                    formula_result = self._extract_formula_stable(
                        formula_name, scored_chunks, text
                    )
                    
                    if formula_result:
                        extracted_formulas.append(formula_result)
                        self.failure_count = 0  # Reset failure count on success
                    else:
                        self.failure_count += 1
                    
                    # Progressive rate limiting based on success/failure
                    if formula_result:
                        time.sleep(0.5)  # Successful extraction
                    else:
                        time.sleep(1.0)  # Failed extraction, wait longer
                
                # Check if emergency stop was triggered
                if self.failure_count >= self.quota_config['emergency_stop_after_failures']:
                    break
                    
                # Inter-batch delay to manage quota
                if batch_end < len(self.target_outputs):
                    time.sleep(self.quota_config['retry_delay'])

            progress_bar.progress(100)
            
            overall_conf = (
                sum(f.confidence for f in extracted_formulas) / len(extracted_formulas) 
                if extracted_formulas else 0.0
            )


            return DocumentExtractionResult(
                input_variables=self.input_variables,
                basic_derived_formulas=self.basic_derived,
                extracted_formulas=extracted_formulas,
                extraction_summary=f"Stable chunked analysis complete. Successfully identified {len(extracted_formulas)} formulas from {len(chunks)} stable chunks using {self.config['chunk_size']} char chunks with {self.config['chunk_overlap']} overlap.",
                overall_confidence=overall_conf,
            )
            
        except Exception as e:
            st.error(f"❌ **Stable chunked extraction failed:** {e}")
            st.exception(e)
            return self._explain_no_extraction()

    def _check_api_status(self) -> bool:
        """Check if API is accessible with a minimal request"""
        try:
            response = client.chat.completions.create(
                model=DEPLOYMENT_NAME,
                messages=[{"role": "user", "content": "Test"}],
            )
            return True
        except Exception as e:
            st.warning(f"API check failed: {e}")
            return False

    def _fallback_to_offline_extraction(self, text: str) -> DocumentExtractionResult:
        """Complete offline extraction when API is unavailable"""
        
        st.info("🔄 Falling back to offline pattern-based extraction...")
        
        extracted_formulas = []
        
        for formula_name in self.target_outputs:
            offline_result = self._extract_formula_offline(formula_name, text)
            if offline_result:
                extracted_formulas.append(offline_result)
        
        overall_conf = (
            sum(f.confidence for f in extracted_formulas) / len(extracted_formulas) 
            if extracted_formulas else 0.0
        )
        
        return DocumentExtractionResult(
            input_variables=self.input_variables,
            basic_derived_formulas=self.basic_derived,
            extracted_formulas=extracted_formulas,
            extraction_summary=f"Offline pattern-based extraction complete. Found {len(extracted_formulas)} formulas using pattern matching (API unavailable).",
            overall_confidence=overall_conf,
        )

    def _create_stable_chunks(self, text: str) -> List[Dict]:
        """Create stable chunks with consistent ratios"""
        
        # Split text using stable configuration
        text_chunks = self.text_splitter.split_text(text)
        
        stable_chunks = []
        for i, chunk_text in enumerate(text_chunks):
            # Only include chunks that meet minimum size requirement
            if len(chunk_text) >= self.config['min_chunk_size']:
                chunk_data = {
                    'id': i,
                    'text': chunk_text,
                    'char_count': len(chunk_text),
                    'word_count': len(chunk_text.split()),
                    'relevance_score': 0.0,  # Will be calculated later
                    'chunk_ratio': len(chunk_text) / len(text),  # Stable ratio
                    'position_ratio': i / len(text_chunks),  # Position in document
                }
                stable_chunks.append(chunk_data)
        
        return stable_chunks

    def _score_chunks_for_relevance(self, chunks: List[Dict]) -> List[Dict]:
        """Score chunks for relevance using stable metrics"""
        
        for chunk in chunks:
            text_lower = chunk['text'].lower()
            score = 0.0
            
            # High priority keywords (weight: 3)
            for keyword in self.formula_keywords['high_priority']:
                score += text_lower.count(keyword) * 3
            
            # Medium priority keywords (weight: 2)
            for keyword in self.formula_keywords['medium_priority']:
                score += text_lower.count(keyword) * 2
            
            # Low priority keywords (weight: 1)
            for keyword in self.formula_keywords['low_priority']:
                score += text_lower.count(keyword) * 1
            
            # Normalize score by chunk length
            normalized_score = score / (chunk['word_count'] / 100)
            
            # Apply position bonus (earlier chunks often contain definitions)
            position_bonus = 1.0 - (chunk['position_ratio'] * 0.3)
            
            chunk['relevance_score'] = normalized_score * position_bonus
        
        # Sort by relevance score (highest first)
        return sorted(chunks, key=lambda x: x['relevance_score'], reverse=True)

    def _extract_formula_stable(self, formula_name: str, scored_chunks: List[Dict], full_text: str) -> Optional[ExtractedFormula]:
        """Extract formula using stable chunking approach"""
        
        # Select top relevant chunks for this formula
        relevant_chunks = self._select_relevant_chunks_for_formula(
            formula_name, scored_chunks
        )
        
        if not relevant_chunks:
            return None
        
        # Combine selected chunks with stable context length
        combined_context = self._combine_chunks_stable(relevant_chunks)
        
        # Extract formula using focused prompt
        return self._extract_formula_with_context(formula_name, combined_context)

    def _select_relevant_chunks_for_formula(self, formula_name: str, scored_chunks: List[Dict]) -> List[Dict]:
        """Select most relevant chunks for specific formula"""
        
        formula_specific_keywords = {
            'surrender': ['surrender', 'gsv', 'ssv', 'cash', 'quit', 'capital units', 'elapsed', 'policy term', '1.05', 'three years', 'redemption'],
            'gsv':['gsv', 'ssv', 'gsv_factor'],
            'surrender_charge': ['surrender charge', 'capital units', '1.05', 'elapsed policy duration', 'policy term', 'three years', 'redemption'],
            'premium': ['premium', 'payment', 'annual', 'monthly'],
            'benefit': ['benefit', 'payout', 'income', 'amount'],
            'death': ['death', 'mortality', 'sum assured'],
            'maturity': ['maturity', 'endowment', 'maturity date'],
            'charge': ['charge', 'fee', 'deduction', 'cost']
        }
        
        # Find relevant keyword category
        relevant_keywords = []
        for category, keywords in formula_specific_keywords.items():
            if any(keyword in formula_name.lower() for keyword in keywords):
                relevant_keywords.extend(keywords)
        
        # Score chunks specifically for this formula
        formula_scored_chunks = []
        for chunk in scored_chunks:
            text_lower = chunk['text'].lower()
            formula_score = chunk['relevance_score']  # Base score
            
            # Add formula-specific scoring
            for keyword in relevant_keywords:
                if keyword in text_lower:
                    formula_score += 2.0
            
            # Check for formula name mentions
            if formula_name.lower() in text_lower:
                formula_score += 5.0
            
            chunk_copy = chunk.copy()
            chunk_copy['formula_score'] = formula_score
            formula_scored_chunks.append(chunk_copy)
        
        # Sort by formula-specific score
        formula_scored_chunks.sort(key=lambda x: x['formula_score'], reverse=True)
        
        # Select top chunks that meet threshold
        selected_chunks = []
        for chunk in formula_scored_chunks:
            if (len(selected_chunks) < self.config['max_chunks_per_formula'] and 
                chunk['formula_score'] >= self.config['relevance_threshold']):
                selected_chunks.append(chunk)
        
        return selected_chunks

    def _combine_chunks_stable(self, chunks: List[Dict]) -> str:
        """Combine chunks with stable context length management"""
        
        combined_text = ""
        current_length = 0
        max_length = self.config['max_context_length']
        
        for chunk in chunks:
            chunk_text = chunk['text']
            
            # Check if adding this chunk would exceed max length
            if current_length + len(chunk_text) > max_length:
                # Truncate the chunk to fit
                remaining_space = max_length - current_length
                if remaining_space > 200:  # Only add if meaningful space left
                    chunk_text = chunk_text[:remaining_space] + "..."
                    combined_text += f"\n\n--- Chunk {chunk['id']} ---\n{chunk_text}"
                break
            
            combined_text += f"\n\n--- Chunk {chunk['id']} ---\n{chunk_text}"
            current_length += len(chunk_text)
        
        return combined_text

    def _extract_formula_with_context(self, formula_name: str, context: str) -> Optional[ExtractedFormula]:
        """Extract formula using stable context and focused prompt with fallback options"""
        
        # List of models to try in order (cheapest to most expensive)
        models_to_try = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4"]
        
        prompt = f"""
        Extract the calculation formula for "{formula_name}" from the following document content.

        DOCUMENT CONTENT:
        {context}

        AVAILABLE VARIABLES:
        {', '.join(self.input_variables.keys())}
    
        INSTRUCTIONS:
        1. Identify a mathematical formula or calculation method for "{formula_name}"
        2. Use the available variables where possible. If others are needed, explain why.
        3. Extract the formula expression as accurately as possible, especially for GSV and Surrender.
        4. Include special conditions or multi-step logic if present               
        5. Pay close attention to formulas involving:
        - Surrender value or GSV
        - exponential terms like (1/1.05)^N
        - conditions like policy term > 3 years
        - Capital Units references or on death mentions

        RESPONSE FORMAT:
        FORMULA_EXPRESSION: [mathematical expression using available variables]

        EXAMPLE:
        If the text says: 
        > "The surrender value payable will be the higher of the guaranteed surrender value (GSV) or the special surrender value (SSV). 
        You would output:
        FORMULA_EXPRESSION: SURRENDER_PAID_AMOUNT= max(GSV, SSV)
        Respond with only the requested format.If unsure about any part, explain why in the response.
        """
        
        for model in models_to_try:
            try:
                response = client.chat.completions.create(
                    model=DEPLOYMENT_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=600,
                    temperature=0.1,
                    top_p=0.95
                )
                
                response_text = response.choices[0].message.content
                
                if "FORMULA_NOT_FOUND" in response_text:
                    return None
                
                return self._parse_stable_formula_response(response_text, formula_name)
                
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "quota" in error_msg.lower():
                    st.warning(f"⚠️ Quota/rate limit for {model}. Trying fallback...")
                    time.sleep(2)  # Brief pause before trying next model
                    continue
                elif "404" in error_msg or "model" in error_msg.lower():
                    st.warning(f"⚠️ Model {model} not available. Trying fallback...")
                    continue
                else:
                    st.error(f"❌ Error with {model} for {formula_name}: {e}")
                    continue
        
        # If all models fail, try offline extraction
        st.warning(f"⚠️ All API models failed for {formula_name}. Attempting offline extraction...")
        return self._extract_formula_offline(formula_name, context)

    def _parse_stable_formula_response(self, response_text: str, formula_name: str) -> Optional[ExtractedFormula]:
        """Parse formula response with stable extraction"""
        
        try:
            # Extract formula expression
            formula_match = re.search(r'FORMULA_EXPRESSION:\s*(.+?)(?=\nVARIABLES_USED|$)', response_text, re.DOTALL | re.IGNORECASE)
            formula_expression = formula_match.group(1).strip() if formula_match else "Formula not clearly defined"
            
            # Extract variables used
            variables_match = re.search(r'VARIABLES_USED:\s*(.+?)(?=\nDOCUMENT_EVIDENCE|$)', response_text, re.DOTALL | re.IGNORECASE)
            variables_str = variables_match.group(1).strip() if variables_match else ""
            specific_variables = self._parse_variables_stable(variables_str)
            
            # Extract document evidence
            evidence_match = re.search(r'DOCUMENT_EVIDENCE:\s*(.+?)(?=\nBUSINESS_CONTEXT|$)', response_text, re.DOTALL | re.IGNORECASE)
            document_evidence = evidence_match.group(1).strip() if evidence_match else "No supporting evidence found"
            
            # Extract business context
            context_match = re.search(r'BUSINESS_CONTEXT:\s*(.+?)(?=\nCONFIDENCE_LEVEL|$)', response_text, re.DOTALL | re.IGNORECASE)
            business_context = context_match.group(1).strip() if context_match else f"Calculation for {formula_name}"
            
            # Extract confidence level
            confidence_match = re.search(r'CONFIDENCE_LEVEL:\s*([0-9]*\.?[0-9]+)', response_text, re.IGNORECASE)
            confidence = float(confidence_match.group(1)) if confidence_match else 0.4
            
            return ExtractedFormula(
                formula_name=formula_name.upper(),
                formula_expression=formula_expression,
                variants_info="Extracted using stable chunking approach",
                business_context=business_context,
                confidence=min(confidence, 1.0),  # Cap at 1.0
                source_method='stable_chunked_extraction',
                document_evidence=document_evidence[:500],  # Limit evidence length
                specific_variables=specific_variables
            )
            
        except Exception as e:
            st.error(f"Error parsing response for {formula_name}: {e}")
            return None

    def _parse_variables_stable(self, variables_str: str) -> Dict[str, str]:
        """Parse variables with stable approach"""
        
        specific_variables = {}
        if variables_str:
            # Clean and split variables
            var_names = [var.strip().upper() for var in variables_str.split(',')]
            
            for var_name in var_names:
                if var_name in self.input_variables:
                    specific_variables[var_name] = self.input_variables[var_name]
                elif var_name in self.basic_derived:
                    specific_variables[var_name] = self.basic_derived[var_name]
                else:
                    # Try to find partial matches
                    for input_var in self.input_variables:
                        if var_name in input_var or input_var in var_name:
                            specific_variables[input_var] = self.input_variables[input_var]
                            break
        
        return specific_variables

    def _explain_no_extraction(self) -> DocumentExtractionResult:
        """Explain that extraction cannot be performed without API key"""
        return DocumentExtractionResult(
            input_variables=self.input_variables,
            basic_derived_formulas=self.basic_derived,
            extracted_formulas=[],
            extraction_summary="Cannot extract formulas from document. A valid `OPENAI_API_KEY` is required to enable stable chunked document analysis and formula extraction.",
            overall_confidence=0.0,
        )


def extract_text_from_file(file_bytes, file_extension):
    """Extract text from supported file formats with better error handling"""
    try:
        if file_extension == '.pdf':
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                tmp_file.write(file_bytes)
                tmp_file_path = tmp_file.name
            
            try:
                text = extract_text_from_pdf_lib(tmp_file_path)
                os.unlink(tmp_file_path)
                
               
                return text
            except Exception as e:
                os.unlink(tmp_file_path)
                raise e
        
        elif file_extension == '.txt':
            text = file_bytes.decode('utf-8')
            if len(text) > 50000:
                st.info(f"📊 Text file size: {len(text)} characters. Using stable chunking.")
            return text
        
        elif file_extension == '.docx':
            try:
                import docx
                from io import BytesIO
                doc = docx.Document(BytesIO(file_bytes))
                text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
                if len(text) > 50000:
                    st.info(f"📊 Word document size: {len(text)} characters. Using stable chunking.")
                return text
            except ImportError:
                st.error("`python-docx` not installed. Please install it: `pip install python-docx`")
                return ""
        
        else:
            return ""
            
    except Exception as e:
        st.error(f"Error extracting text from file: {e}")
        return ""

# Usage example:
# extractor = StableChunkedDocumentFormulaExtractor(DEFAULT_TARGET_OUTPUT_VARIABLES)
# result = extractor.extract_formulas_from_document(document_text)

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
        input:focus, select:focus, textarea:focus {
        outline: none !important;
        box-shadow: none !important;
        border: 1px solid #ccc !important;
    }

    /* Remove red border for invalid inputs */
    input:invalid, select:invalid {
        border: 1px solid #ccc !important;
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
        h2, h3, h4, h5, h6 {
            font-family: 'Montserrat', sans-serif !important;
            color: #004DA8 !important;
            font-weight: 700;
            margin-top: 1.8em;
            margin-bottom: 0.8em;
            text-shadow: 1px 1px 2px rgba(0, 77, 168, 0.1);
            animation: slideIn 0.8s ease-out;
        }
        h1{
            font-family: 'Montserrat', sans-serif !important;
            color: #004DA8 !important;
            font-size: 2.5rem;
            font-weight: 750;
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
            color: #14212e !important;
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

    

    /* Make sure Streamlit content is not covered */
    .main > div {
        padding-top: 20px;
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
            color: #2f3b4a !important;
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

    # Apply custom CSS
    set_custom_css()

    # Custom header bar with logo and title
    st.markdown(
        """
        <style>
            .header-container {
                padding: 1rem 0;
            }

            .header-bar {
                display: flex;
                align-items: center;
                gap: 1rem;
                background-color: white;
                padding: 1rem;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            }

            .header-title {
                font-size: 2.5rem;
                font-weight: 750;
                color: #004DA8 !important;
                font-family: 'Segoe UI', sans-serif;
                margin: 0;
            }

            .header-bar img {
                height: 100px;
            }

            .formula-table {
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                overflow: hidden;
                background-color: white;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                margin-bottom: 1rem;
            }

            .formula-table-header {
                background-color: #f8f9fa;
                border-bottom: 2px solid #e0e0e0;
                padding: 0.75rem;
                font-weight: 600;
                font-size: 1.1rem;
                color: #004DA8;
            }

            .formula-row {
                padding: 0.5rem 0.75rem;
                border-bottom: 1px solid #e0e0e0;
            }

            .formula-row:last-child {
                border-bottom: none;
            }

            .formula-row:hover {
                background-color: #f8f9fa;
            }

            .add-formula-section {
                margin-top: 1rem;
                padding: 1rem;
                background-color: transparent;
                border: none;
            }

            .add-formula-title {
                font-size: 1.2rem;
                font-weight: 600;
                color: #004DA8;
                margin-bottom: 1rem;
            }

            .section-divider {
                margin: 2rem 0;
                border-bottom: 1px solid #e0e0e0;
            }
        </style>

        <div class="header-container">
            <div class="header-bar">
                <img src="https://raw.githubusercontent.com/AyushiR0y/streamlit_formulagen/main/assets/logo.png">
                <div class="header-title">Document Formula Extractor</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


    st.set_page_config(
        page_title="Document Formula Extractor",
        page_icon="https://github.com/AyushiR0y/streamlit_formulagen/raw/64b69f5e22fdd673d9ae58fdee24700687b372c1/assets/Dragnfly.png",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state
    if 'extraction_result' not in st.session_state:
        st.session_state.extraction_result = None
    if 'selected_output_variables' not in st.session_state:
        st.session_state.selected_output_variables = DEFAULT_TARGET_OUTPUT_VARIABLES.copy()
    if 'custom_output_variable' not in st.session_state:
        st.session_state.custom_output_variable = ""
    if 'user_defined_output_variables' not in st.session_state:
        st.session_state.user_defined_output_variables = []
    if 'formulas' not in st.session_state:
        st.session_state.formulas = []
    if 'formulas_saved' not in st.session_state:
        st.session_state.formulas_saved = False
    if 'editing_formula' not in st.session_state:
        st.session_state.editing_formula = -1  # -1 means no formula is being edited

    # Track when keywords change to reset file upload
    if 'previous_selected_variables' not in st.session_state:
        st.session_state.previous_selected_variables = DEFAULT_TARGET_OUTPUT_VARIABLES.copy()

    # Main Content Area
    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        # Output Variable Selection
        st.subheader("Select Keywords")
        st.markdown("Choose the keywords you want to extract from the document to generate formulas. You can also add custom ones!")

        # Combine default and user-defined variables for the multiselect options
        all_possible_output_variables = sorted(list(set(DEFAULT_TARGET_OUTPUT_VARIABLES + st.session_state.user_defined_output_variables)))

        # Update selected_output_variables with the multiselect
        current_selection = st.multiselect(
            "Target Keywords:",
            options=all_possible_output_variables,
            default=st.session_state.selected_output_variables,
            help="These are the keywords for which the system will try to find formulas. Select all that apply."
        )

        # Check if selection changed - if so, reset file upload and extraction results
        if current_selection != st.session_state.previous_selected_variables:
            st.session_state.selected_output_variables = current_selection
            st.session_state.previous_selected_variables = current_selection.copy()
            st.session_state.extraction_result = None
            st.session_state.formulas = []
            st.session_state.formulas_saved = False
            st.session_state.editing_formula = -1
            # Force rerun to clear file uploader
            st.rerun()
        else:
            st.session_state.selected_output_variables = current_selection

        # Add custom output variable
        st.session_state.custom_output_variable = st.text_input(
            "Add a custom keyword (e.g., 'DEATH_BENEFIT_CALC'):",
            value=st.session_state.custom_output_variable,
            key="custom_output_input",
            help="Enter a specific keyword whose formulas you want to extract, even if it's not in the default list. Press 'Add' to include it."
        )

        if st.button("Add Custom Keyword", key="add_custom_formula_button"):
            new_var = st.session_state.custom_output_variable.strip()
            if new_var and new_var not in st.session_state.user_defined_output_variables and new_var not in DEFAULT_TARGET_OUTPUT_VARIABLES:
                st.session_state.user_defined_output_variables.append(new_var)
                if new_var not in st.session_state.selected_output_variables:
                    st.session_state.selected_output_variables.append(new_var)
                    st.session_state.previous_selected_variables = st.session_state.selected_output_variables.copy()
                st.session_state.custom_output_variable = ""
                # Clear previous results when adding new custom variable
                st.session_state.extraction_result = None
                st.session_state.formulas = []
                st.session_state.formulas_saved = False
                st.session_state.editing_formula = -1
                st.success(f"'{new_var}' added and selected!")
                st.rerun()
            elif new_var in st.session_state.user_defined_output_variables or new_var in DEFAULT_TARGET_OUTPUT_VARIABLES:
                st.info(f"'{new_var}' is already in the list of available formulas.")
            else:
                st.warning("Please enter a valid custom formula name to add.")

        st.markdown("---")

        st.subheader("Upload Product Specifications")
        st.markdown("Upload your insurance policy document (PDF, TXT, DOCX) to begin formula extraction.")

        # Generate a unique key based on selected variables to force file uploader reset
        file_uploader_key = f"file_uploader_{hash(str(sorted(st.session_state.selected_output_variables)))}"
        
        uploaded_file = st.file_uploader(
            "Select a document",
            type=list(ALLOWED_EXTENSIONS),
            help=f"Accepts: {', '.join(ALLOWED_EXTENSIONS)}. Max file size: {MAX_FILE_SIZE / (1024*1024):.1f} MB",
            key=file_uploader_key
        )

        # Rest of your file processing code remains the same...
        if uploaded_file is not None:
            if uploaded_file.size > MAX_FILE_SIZE:
                st.error(f"File size exceeds the limit. Please upload a file smaller than {MAX_FILE_SIZE / (1024*1024):.1f} MB.")
                st.session_state.extraction_result = None
            else:
                st.info(f"**File Selected:** `{uploaded_file.name}` (`{uploaded_file.size / 1024:.1f} KB`)")

                if st.button("Analyze Document", type="primary", key="analyze_button"):
                    if not st.session_state.selected_output_variables:
                        st.warning("Please select at least one target formula to extract or add a custom one.")
                        st.session_state.extraction_result = None
                    else:
                        with st.spinner("Analyzing document and extracting formulas... This may take a moment."):
                            file_extension = Path(uploaded_file.name).suffix.lower()
                            text = extract_text_from_file(uploaded_file.read(), file_extension)

                            if not text.strip():
                                st.error("Could not extract readable text from the uploaded file. Please ensure it contains text content.")
                                st.session_state.extraction_result = None
                            else:
                                # Only proceed with extraction if API key is configured
                                if not MOCK_MODE and AZURE_API_KEY:
                                    extractor = StableChunkedDocumentFormulaExtractor(target_outputs=st.session_state.selected_output_variables)
                                    extraction_result = extractor.extract_formulas_from_document(text)
                                                            
                                    st.session_state.extraction_result = extraction_result
                                    # Convert to editable format
                                    st.session_state.formulas = [
                                        {
                                            "formula_name": formula.formula_name,
                                            "formula_expression": formula.formula_expression
                                        } for formula in extraction_result.extracted_formulas
                                    ]
                                    st.session_state.formulas_saved = False
                                    st.session_state.editing_formula = -1  # Reset editing state
                                else:
                                    st.session_state.extraction_result = None
                                    st.warning("Cannot perform extraction without a configured OPENAI_API_KEY.")
        else:
            st.session_state.extraction_result = None
    with col2:
        st.subheader("Reference Variables")
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
            current_targets = sorted(list(set(st.session_state.selected_output_variables)))
            if current_targets:
                target_data = [{"Target Variable": var} for var in current_targets]
                st.dataframe(pd.DataFrame(target_data), use_container_width=True, hide_index=True)
            else:
                st.info("No target formulas selected yet. Use the multiselect or add custom formulas on the left.")

    st.markdown("---")

    # Display Results
    if st.session_state.extraction_result:
        st.subheader("Extraction Summary")

        result = st.session_state.extraction_result

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric(label="Total Formulas Found", value=len(st.session_state.formulas), help="Number of distinct formulas successfully extracted.")
        with col_s2:
            st.metric(label="Overall Confidence", value=f"{result.overall_confidence:.1%}", help="Average confidence score across all extracted formulas.")

        st.info(result.extraction_summary)

        if st.session_state.formulas:
            st.subheader("Formula Overview")
            st.markdown("Review and edit the extracted formulas. You can modify expressions, delete formulas, or add new ones.")

            # Create table header
            col_header1, col_header2, col_header3 = st.columns([3, 5, 2])
            with col_header1:
                st.markdown("**Formula Name**")
            with col_header2:
                st.markdown("**Expression**")
            with col_header3:
                st.markdown("**Actions**")

            # Display existing formulas with improved layout
            for i, formula in enumerate(st.session_state.formulas):
                col1, col2, col3 = st.columns([3, 5, 2])
                
                with col1:
                    if st.session_state.editing_formula == i:
                        # Editable formula name
                        new_name = st.text_input(
                            "Formula Name",
                            value=formula.get("formula_name", ""),
                            key=f"name_{i}",
                            label_visibility="collapsed"
                        )
                    else:
                        # Display only
                        st.markdown(f"**{formula.get('formula_name', '')}**")
                    
                with col2:
                    if st.session_state.editing_formula == i:
                        # Editable formula expression
                        new_expression = st.text_area(
                            "Expression",
                            value=formula.get("formula_expression", ""),
                            key=f"expr_{i}",
                            label_visibility="collapsed",
                            height=68
                        )
                    else:
                        # Display only
                        st.code(formula.get("formula_expression", ""), language="python")
                    
                with col3:
                    if st.session_state.editing_formula == i:
                        # Save and Cancel buttons when editing
                        col_save, col_cancel = st.columns(2)
                        with col_save:
                            if st.button("Save", key=f"save_{i}", help="Save changes"):
                                st.session_state.formulas[i]["formula_name"] = new_name
                                st.session_state.formulas[i]["formula_expression"] = new_expression
                                st.session_state.formulas_saved = False
                                st.session_state.editing_formula = -1
                                st.success("Formula updated!")
                                st.rerun()
                        
                        with col_cancel:
                            if st.button("Cancel", key=f"cancel_{i}", help="Cancel editing"):
                                st.session_state.editing_formula = -1
                                st.rerun()
                    else:
                        # Edit and Delete buttons when not editing
                        col_edit, col_delete = st.columns(2)
                        with col_edit:
                            if st.button("Edit", key=f"edit_{i}", help="Edit this formula"):
                                st.session_state.editing_formula = i
                                st.rerun()
                        
                        with col_delete:
                            if st.button("Delete", key=f"delete_{i}", help="Delete this formula"):
                                st.session_state.formulas.pop(i)
                                st.session_state.formulas_saved = False
                                st.session_state.editing_formula = -1
                                st.success("Formula deleted!")
                                st.rerun()
                
                # Add separator between rows
                if i < len(st.session_state.formulas) - 1:
                    st.markdown('<hr style="margin: 0.5rem 0; border: 0; border-top: 1px solid #e0e0e0;">', unsafe_allow_html=True)

            # Section divider after table
            st.markdown("---")

            # Add new formula section
            st.markdown("#### Add New Formula")
            col_add1, col_add2, col_add3 = st.columns([3, 5, 2])
            
            with col_add1:
                new_formula_name = st.text_input("New Formula Name", key="new_formula_name", label_visibility="collapsed", placeholder="Enter formula name")
            
            with col_add2:
                new_formula_expression = st.text_area("New Formula Expression", key="new_formula_expression", label_visibility="collapsed", placeholder="Enter formula expression", height=68)
            
            with col_add3:
                st.write("")  # Space for alignment
                if st.button("Add Formula", key="add_new_formula", help="Add new formula"):
                    if new_formula_name.strip() and new_formula_expression.strip():
                        new_formula = {
                            "formula_name": new_formula_name.strip(),
                            "formula_expression": new_formula_expression.strip()
                        }
                        st.session_state.formulas.append(new_formula)
                        st.session_state.formulas_saved = False
                        st.success(f"Added formula: {new_formula_name}")
                        st.rerun()
                    else:
                        st.warning("Please enter both formula name and expression.")

            st.markdown("---")

            # Save and Reset buttons
            col_save1, col_save2 = st.columns([1, 1])
            
            with col_save1:
                if st.button("Save All Changes", type="primary", key="save_formulas"):
                    if st.session_state.formulas:
                        st.session_state.formulas_saved = True
                        st.session_state.editing_formula = -1
                        st.success("All formulas saved! Changes will be reflected in downloads.")
                    else:
                        st.warning("No formulas to save.")
            
            with col_save2:
                if st.button("Reset All Formulas", key="reset_formulas"):
                    st.session_state.formulas = []
                    st.session_state.formulas_saved = False
                    st.session_state.editing_formula = -1
                    st.success("All formulas reset.")
                    st.rerun()

            st.markdown("---")

            st.subheader("Export Extracted Data")
            st.markdown("Download the extracted formulas in various formats for further analysis or integration.")

            # Create export data from current session state (user-edited formulas)
            export_data = {
                "extraction_summary": result.extraction_summary,
                "total_formulas": len(st.session_state.formulas),
                "overall_confidence": result.overall_confidence,
                "formulas": st.session_state.formulas
            }

            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                st.download_button(
                    label="Download as JSON",
                    data=json.dumps(export_data, indent=2),
                    file_name="extracted_formulas.json",
                    mime="application/json",
                    help="Export all detailed formula data in JSON format."
                )

            with col_exp2:
                # Create CSV data from current formulas (user-edited)
                csv_data = pd.DataFrame([
                    {
                        "Formula Name": f.get("formula_name", ""),
                        "Expression": f.get("formula_expression", "")
                    } for f in st.session_state.formulas
                ]).to_csv(index=False)
                
                st.download_button(
                    label="Download as CSV",
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
            <p>Developed with love using Streamlit and OpenAI API</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
