import pytest
from app.process import (
    initialize_glossary,
    create_translation_prompt,
    translator
)
from unittest.mock import patch

@pytest.fixture(autouse=True)
def mock_openai():
    with patch('app.patch.get_completion', return_value="Bienvenue au Casino NetBet"), \
         patch('app.patch.MODEL', 'gpt-4'), \
         patch('app.patch.TEMPERATURE', 0.3), \
         patch('app.patch.JS_MODE', False), \
         patch('app.patch.RPM', 3), \
         patch('openai.OpenAI') as mock_client:
        # Mock the OpenAI client response structure
        mock_response = type('obj', (), {
            'choices': [
                type('obj', (), {
                    'message': type('obj', (), {
                        'content': "Bienvenue au Casino NetBet"
                    })
                })
            ]
        })
        mock_client.return_value.chat.completions.create.return_value = mock_response
        yield mock_client

def test_glossary_initialization():
    """Test if glossary processor initializes and loads glossaries correctly"""
    glossary = initialize_glossary()
    print("\nLooking for glossaries in:")
    print("JSON dir:", glossary.json_dir)
    print("CSV dir:", glossary.csv_dir)
    print("\nGlossary files loaded:", glossary.glossaries.keys())
    print("\nGlossary contents:", glossary.glossaries)
    assert glossary.glossaries is not None
    assert len(glossary.glossaries) > 0  # Should have loaded the NetBet glossaries

def test_term_identification():
    """Test if terms are correctly identified from the text"""
    processor = initialize_glossary()
    test_text = "Welcome to NetBet Casino"
    terms = processor.identify_terms(test_text, "en", "fr")
    
    # Print debug info
    print("\nTest text:", test_text)
    print("Identified terms:", terms)
    print("Available glossaries:", processor.glossaries.keys())
    
    # Check if "Casino" is in the terms
    assert "Casino" in terms or "NetBet Casino" in terms, "Expected casino-related terms not found"

def test_term_marking():
    """Test if terms are correctly marked in the text"""
    processor = initialize_glossary()
    test_text = "Welcome to NetBet Casino"
    terms = processor.identify_terms(test_text, "en", "fr")
    marked_text = processor.mark_terms(test_text, terms)
    
    # Should contain [[term]] markers
    assert "[[" in marked_text and "]]" in marked_text

def test_translation_prompt_creation():
    """Test if translation prompt is created correctly with glossary terms"""
    processor = initialize_glossary()
    test_text = "Welcome to NetBet Casino"
    terms = {"Casino": "Casino", "NetBet": "NetBet"}  # Provide known terms
    marked_text = processor.mark_terms(test_text, terms)
    
    prompt = create_translation_prompt(marked_text, terms, "en", "fr")
    
    # Print debug info
    print("\nGenerated prompt:", prompt)
    print("Terms:", terms)
    
    assert test_text in prompt, "Original text not found in prompt"
    assert "glossary" in prompt.lower(), "Glossary section not found in prompt"
    assert "Casino → Casino" in prompt, "Term translation not found in prompt"

def test_full_translation_flow():
    """Test the complete translation process with glossary terms"""
    test_text = "Welcome to NetBet Casino. Enjoy our Live Casino games."
    
    result = translator(
        source_lang="en",
        target_lang="fr",
        source_text=test_text,
        tone=3,
        country="FR",
        max_tokens=1000
    )
    
    assert result is not None
    assert len(result) > 0
    
    # Check if key terms are preserved in translation
    processor = initialize_glossary()
    terms = processor.identify_terms(test_text, "en", "fr")
    success, issues = processor.validate_translation(test_text, result, terms)
    
    # Optional: Print issues for debugging
    if not success:
        print(f"Translation issues found: {issues}")

def test_long_text_translation():
    """Test translation with a longer text that requires chunking"""
    # Create a longer text that exceeds the token limit
    long_text = " ".join(["Welcome to NetBet Casino"] * 50)
    
    result = translator(
        source_lang="en",
        target_lang="fr",
        source_text=long_text,
        tone=3,
        country="FR",
        max_tokens=1000
    )
    
    assert result is not None
    assert len(result) > 0
    
    # Verify glossary terms are preserved even in long translations
    processor = initialize_glossary()
    terms = processor.identify_terms(long_text, "en", "fr")
    success, issues = processor.validate_translation(long_text, result, terms)
    
    if not success:
        print(f"Long text translation issues: {issues}")

if __name__ == "__main__":
    pytest.main([__file__]) 