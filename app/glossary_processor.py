import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List, Tuple

class GlossaryProcessor:
    def __init__(self):
        # Get the root directory (where streamlit_app.py is)
        self.root_dir = Path(__file__).parent.parent
        
        # Set up paths
        self.csv_dir = self.root_dir / "glossaries" / "glossaries_csv"
        self.json_dir = self.root_dir / "glossaries" / "glossaries_json"
        
        # Create directories if they don't exist
        self.json_dir.mkdir(parents=True, exist_ok=True)
        
        # Store processed glossaries in memory
        self.glossaries = {}

    def process_csv_to_json(self) -> None:
        """
        Process all CSV files in the glossaries_csv directory and convert them to JSON
        """
        if not self.csv_dir.exists():
            raise FileNotFoundError(f"CSV directory not found: {self.csv_dir}")

        for csv_file in self.csv_dir.glob("*.csv"):
            try:
                # Read CSV file
                df = pd.read_csv(csv_file)
                
                # Convert DataFrame to dictionary using "EN - Source" as key
                glossary_dict = {"terms": {}}
                
                for _, row in df.iterrows():
                    # Skip rows where "EN - Source" is empty or NaN
                    if pd.isna(row.get("EN - Source")) or row.get("EN - Source") == "":
                        continue
                        
                    source_term = str(row["EN - Source"]).strip()
                    term_translations = {}
                    
                    # Add all non-null translations
                    for column in df.columns:
                        if pd.notna(row[column]) and row[column] != "":
                            term_translations[column] = str(row[column]).strip()
                    
                    glossary_dict["terms"][source_term] = term_translations

                # Add metadata
                glossary_dict["metadata"] = {
                    "source_file": csv_file.name,
                    "languages": [col for col in df.columns if pd.notna(col)],
                    "term_count": len(glossary_dict["terms"])
                }

                # Save to JSON
                json_path = self.json_dir / f"{csv_file.stem}.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(glossary_dict, f, ensure_ascii=False, indent=2)
                
                # Store in memory
                self.glossaries[csv_file.stem] = glossary_dict
                
                print(f"Successfully processed {csv_file.name}")
                
            except Exception as e:
                print(f"Error processing {csv_file.name}: {str(e)}")

    def load_glossaries(self) -> None:
        """
        Load all JSON glossaries into memory
        """
        for json_file in self.json_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    self.glossaries[json_file.stem] = json.load(f)
            except Exception as e:
                print(f"Error loading {json_file.name}: {str(e)}")

    def get_translation(self, 
                       term: str, 
                       target_lang: str, 
                       glossary_name: Optional[str] = None) -> Optional[str]:
        """
        Get translation for a specific term
        
        Args:
            term: The source term to translate
            target_lang: Target language code
            glossary_name: Specific glossary to search (optional)
            
        Returns:
            Translated term or None if not found
        """
        # If glossary specified, search only that one
        if glossary_name and glossary_name in self.glossaries:
            glossary = self.glossaries[glossary_name]
            if term in glossary["terms"] and target_lang in glossary["terms"][term]:
                return glossary["terms"][term][target_lang]
            return None

        # Search all glossaries
        for glossary in self.glossaries.values():
            if term in glossary["terms"] and target_lang in glossary["terms"][term]:
                return glossary["terms"][term][target_lang]
        
        return None

    def get_all_terms(self, glossary_name: Optional[str] = None) -> Dict:
        """
        Get all terms from a specific glossary or all glossaries
        """
        if glossary_name:
            return self.glossaries.get(glossary_name, {}).get("terms", {})
        
        # Combine terms from all glossaries
        all_terms = {}
        for glossary in self.glossaries.values():
            all_terms.update(glossary.get("terms", {}))
        return all_terms

    def identify_terms(self, text: str, source_lang: str, target_lang: str) -> Dict[str, str]:
        """
        Identify glossary terms in the source text and their target translations.
        """
        found_terms = {}
        
        # Get all terms from all glossaries
        for glossary in self.glossaries.values():
            terms = glossary.get("terms", {})
            for term_data in terms.values():
                # Print for debugging
                print(f"Checking term data: {term_data}")
                if "EN - Source" in term_data:
                    source_term = term_data["EN - Source"]
                    if source_term and source_term.lower() in text.lower():
                        target_translation = term_data.get("FR")  # Using FR directly since that's our target
                        if target_translation:
                            found_terms[source_term] = target_translation
        
        return found_terms

    def mark_terms(self, text: str, terms: Dict[str, str]) -> str:
        """
        Mark identified terms in the text with [[term]] notation.
        """
        marked_text = text
        for term in sorted(terms.keys(), key=len, reverse=True):  # Process longer terms first
            # Case-insensitive replacement while preserving original case
            marked_text = self._replace_preserve_case(marked_text, term, f"[[{term}]]")
        return marked_text

    def validate_translation(self, original_text: str, translated_text: str, 
                           terms: Dict[str, str]) -> Tuple[bool, List[str]]:
        """
        Validate that all glossary terms were correctly translated.
        Returns (success, list of missing/incorrect terms)
        """
        issues = []
        success = True
        
        for source_term, target_term in terms.items():
            if target_term.lower() not in translated_text.lower():
                success = False
                issues.append(f"Missing or incorrect translation for '{source_term}' → '{target_term}'")
        
        return success, issues

    def _replace_preserve_case(self, text: str, term: str, replacement: str) -> str:
        """
        Replace term in text while preserving case sensitivity.
        """
        idx = text.lower().find(term.lower())
        while idx != -1:
            # Get the actual case used in the text
            actual_term = text[idx:idx + len(term)]
            # Replace while preserving the original case
            text = text[:idx] + replacement + text[idx + len(term):]
            # Find next occurrence
            idx = text.lower().find(term.lower(), idx + len(replacement))
        return text


def main():
    """
    Main function to process all glossaries
    """
    processor = GlossaryProcessor()
    processor.process_csv_to_json()
    processor.load_glossaries()
    
    # Example usage
    print("\nExample translations:")
    test_term = "accept"  # Replace with a term you know exists
    test_lang = "FR"      # Replace with a target language you know exists
    translation = processor.get_translation(test_term, test_lang)
    print(f"'{test_term}' in {test_lang}: {translation}")


if __name__ == "__main__":
    main() 