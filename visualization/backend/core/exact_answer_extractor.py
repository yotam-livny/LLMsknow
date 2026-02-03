"""
Extract exact answer tokens from model outputs.
Based on the paper "LLMs Know More Than They Show" - the exact answer tokens
contain concentrated truthfulness information.
"""
import torch
from typing import List, Dict, Optional, Tuple, Any
from utils.logging import get_logger

logger = get_logger(__name__)


class ExactAnswerExtractor:
    """
    Extracts exact answer tokens from model outputs.
    
    The exact answer tokens are the tokens that contain the substantive answer
    to the question, whether correct or not. These tokens encode truthfulness
    information more strongly than other tokens.
    """
    
    EXTRACTION_PROMPT = """Extract from the following long answer the short answer, only the relevant tokens. If the long answer does not answer the question, output NO ANSWER.

Q: Which musical featured the song The Street Where You Live?
A: The song "The Street Where You Live" is from the Lerner and Loewe musical "My Fair Lady." It is one of the most famous songs from the show, and it is sung by Professor Henry Higgins as he reflects on the transformation of Eliza Doolittle and the memories they have shared together.
Exact answer: My Fair Lady

Q: Which Swedish actress won the Best Supporting Actress Oscar for Murder on the Orient Express?
A: I'm glad you asked about a Swedish actress who won an Oscar for "Murder on the Orient Express," but I must clarify that there seems to be a misunderstanding here. No Swedish actress has won an Oscar for Best Supporting Actress for that film.
Exact answer: NO ANSWER

Q: Who played Terry Benedict in Ocean's Eleven?
A: The character Terry Benedict in the movie "Ocean's Eleven" was acted by actor Andy Garcia. Benedict is the wealthy and powerful casino magnate.
Exact answer: Andy Garcia

Q: {question}
A: {model_answer}
Exact answer:"""

    def __init__(self, model_manager):
        """
        Initialize the extractor with a model manager.
        
        Args:
            model_manager: The ModelManager instance for tokenization and generation
        """
        self.mm = model_manager
    
    def extract_exact_answer(
        self,
        question: str,
        model_answer: str,
        expected_answer: Optional[str] = None,
        is_correct: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Extract exact answer from model output.
        
        If the answer is known to be correct and we have the expected answer,
        we can directly find where it appears. Otherwise, we use the LLM
        to extract the short answer.
        
        Args:
            question: The original question
            model_answer: The model's generated answer
            expected_answer: Optional expected answer for correct answers
            is_correct: Whether the answer is known to be correct
            
        Returns:
            Dict with:
                - exact_answer: The extracted exact answer text
                - start_char: Start character position in model_answer
                - end_char: End character position
                - extraction_method: 'direct' or 'llm'
                - valid: Whether extraction was successful
        """
        # If we know it's correct and have expected answer, use direct matching
        if is_correct and expected_answer:
            result = self._extract_direct(model_answer, expected_answer)
            if result["valid"]:
                return result
        
        # Otherwise use LLM extraction
        return self._extract_with_llm(question, model_answer)
    
    def _extract_direct(
        self,
        model_answer: str,
        expected_answer: str
    ) -> Dict[str, Any]:
        """
        Extract by directly finding the expected answer in the model output.
        """
        # Handle list of acceptable answers
        answers_to_try = [expected_answer]
        try:
            parsed = eval(expected_answer)
            if isinstance(parsed, list):
                answers_to_try = parsed
        except:
            pass
        
        # Find the first occurrence of any acceptable answer
        best_match = None
        best_index = len(model_answer)
        
        for ans in answers_to_try:
            if isinstance(ans, (int, float)):
                ans = str(ans)
            idx = model_answer.lower().find(str(ans).lower())
            if idx != -1 and idx < best_index:
                best_index = idx
                best_match = ans
        
        if best_match is None:
            return {
                "exact_answer": None,
                "start_char": -1,
                "end_char": -1,
                "extraction_method": "direct",
                "valid": False
            }
        
        return {
            "exact_answer": best_match,
            "start_char": best_index,
            "end_char": best_index + len(best_match),
            "extraction_method": "direct",
            "valid": True
        }
    
    def _extract_with_llm(
        self,
        question: str,
        model_answer: str,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Use the LLM to extract the exact answer from the model output.
        """
        prompt = self.EXTRACTION_PROMPT.format(
            question=question,
            model_answer=model_answer
        )
        
        for attempt in range(max_retries):
            try:
                # Tokenize the prompt
                input_ids = self.mm.tokenize(prompt)
                
                # Generate extraction
                output = self.mm.generate(
                    input_ids,
                    max_new_tokens=50,
                    do_sample=(attempt > 0),  # Use sampling on retries
                    temperature=0.7 if attempt > 0 else 1.0
                )
                
                # Decode the response
                generated_ids = output.sequences[0][input_ids.shape[1]:]
                exact_answer = self.mm.decode(generated_ids)
                
                # Clean up the response
                exact_answer = self._clean_extraction(exact_answer)
                
                # Validate - must appear in the answer or be "NO ANSWER"
                if exact_answer == "NO ANSWER":
                    return {
                        "exact_answer": "NO ANSWER",
                        "start_char": -1,
                        "end_char": -1,
                        "extraction_method": "llm",
                        "valid": True
                    }
                
                # Find where it appears
                idx = model_answer.lower().find(exact_answer.lower())
                if idx != -1:
                    return {
                        "exact_answer": exact_answer,
                        "start_char": idx,
                        "end_char": idx + len(exact_answer),
                        "extraction_method": "llm",
                        "valid": True
                    }
                
                logger.debug(f"Extraction attempt {attempt + 1}: '{exact_answer}' not found in answer")
                
            except Exception as e:
                logger.warning(f"Extraction attempt {attempt + 1} failed: {e}")
        
        return {
            "exact_answer": None,
            "start_char": -1,
            "end_char": -1,
            "extraction_method": "llm",
            "valid": False
        }
    
    def _clean_extraction(self, text: str) -> str:
        """Clean up extracted answer text."""
        # Remove common model artifacts
        text = text.replace("</s>", "").replace("<|eot_id|>", "")
        text = text.replace(".<|", "").replace("|>", "")
        
        # Take first line/sentence
        text = text.split('\n')[0]
        text = text.split('(')[0]  # Remove parenthetical notes
        
        # Clean whitespace and punctuation
        text = text.strip().strip(".")
        
        return text
    
    def find_token_positions(
        self,
        full_text: str,
        exact_answer: str,
        start_char: int,
        tokenizer
    ) -> List[int]:
        """
        Find the token positions that correspond to the exact answer.
        
        This maps character positions to token positions in the tokenized sequence.
        
        Args:
            full_text: The full text that was tokenized
            exact_answer: The exact answer substring
            start_char: Start character position of exact answer
            tokenizer: The tokenizer used
            
        Returns:
            List of token indices that contain the exact answer
        """
        if start_char < 0 or not exact_answer:
            return []
        
        # Tokenize the full text
        encoding = tokenizer(full_text, return_offsets_mapping=True)
        offsets = encoding.offset_mapping
        
        end_char = start_char + len(exact_answer)
        
        # Find tokens that overlap with the exact answer span
        token_positions = []
        for idx, (token_start, token_end) in enumerate(offsets):
            if token_start is None or token_end is None:
                continue
            # Check if token overlaps with exact answer span
            if token_end > start_char and token_start < end_char:
                token_positions.append(idx)
        
        logger.debug(f"Exact answer '{exact_answer}' maps to token positions: {token_positions}")
        return token_positions
    
    def get_exact_answer_token_indices_in_output(
        self,
        tokens: List[Dict],
        exact_answer: str,
        generated_answer: str
    ) -> List[int]:
        """
        Find which output token indices correspond to the exact answer.
        
        Args:
            tokens: List of TokenInfo dicts from inference result
            exact_answer: The extracted exact answer
            generated_answer: The full generated answer text
            
        Returns:
            List of token indices (positions) that contain the exact answer
        """
        if not exact_answer or exact_answer == "NO ANSWER":
            return []
        
        # Normalize for comparison
        exact_lower = exact_answer.lower().strip()
        
        # Get only output tokens
        output_tokens = [t for t in tokens if not t.get("is_input", False)]
        
        if not output_tokens:
            return []
        
        # Method 1: Try to find by sliding window of token texts
        # Concatenate consecutive tokens and check if they contain the exact answer
        best_match = None
        best_match_len = float('inf')
        
        for start_idx in range(len(output_tokens)):
            combined_text = ""
            for end_idx in range(start_idx, min(start_idx + 10, len(output_tokens))):  # Max 10 tokens
                combined_text += output_tokens[end_idx]["text"]
                combined_lower = combined_text.lower().strip()
                
                # Check if this span contains the exact answer
                if exact_lower in combined_lower:
                    span_len = end_idx - start_idx + 1
                    # Prefer shorter spans (more precise match)
                    if span_len < best_match_len:
                        best_match = (start_idx, end_idx)
                        best_match_len = span_len
                    break  # Found a match starting at this position
                
                # Also check if exact answer starts with this combined text
                # (for cases where exact answer extends beyond our window)
                if len(combined_lower) > len(exact_lower):
                    break  # Combined text is longer but doesn't contain it
        
        if best_match:
            start_idx, end_idx = best_match
            positions = [output_tokens[i]["position"] for i in range(start_idx, end_idx + 1)]
            logger.debug(f"Exact answer '{exact_answer}' found in tokens -> positions: {positions}")
            return positions
        
        # Method 2: Fallback - find by matching generated_answer with offset tracking
        # This handles cases where token boundaries don't align nicely
        idx = generated_answer.lower().find(exact_lower)
        if idx == -1:
            logger.debug(f"Exact answer '{exact_answer}' not found in generated answer")
            return []
        
        # Track character position as we go through tokens
        # The generated_answer should match the decoded tokens
        char_pos = 0
        matching_positions = []
        end_idx = idx + len(exact_answer)
        
        for token in output_tokens:
            token_text = token["text"]
            token_start = char_pos
            token_end = char_pos + len(token_text)
            
            # Check if this token overlaps with the exact answer span
            if token_end > idx and token_start < end_idx:
                matching_positions.append(token["position"])
            
            char_pos = token_end
        
        if matching_positions:
            logger.debug(f"Exact answer '{exact_answer}' at char [{idx}:{end_idx}] -> positions: {matching_positions}")
        else:
            logger.debug(f"Exact answer '{exact_answer}' could not be mapped to token positions")
        
        return matching_positions


def get_exact_answer_extractor(model_manager) -> ExactAnswerExtractor:
    """Get an ExactAnswerExtractor instance."""
    return ExactAnswerExtractor(model_manager)
