import os
from typing import List, Union

import openai
import tiktoken
from dotenv import load_dotenv
from icecream import ic
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()  # read local .env file
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MAX_TOKENS_PER_CHUNK = (
    1000  # if text is more than this many tokens, we'll break it up into
)
# discrete chunks to translate one chunk at a time

tone_mapping = {
    1: "Use very informal and casual language, similar to everyday speech.",
    2: "Use informal but polite language, appropriate for casual conversation.",
    3: "Use neutral and professional language, suitable for general business communication.",
    4: "Use formal and business-oriented language, appropriate for official documents.",
    5: "Use very formal and highly structured language, as required for legal or academic writing."
}


def get_completion(
    prompt: str,
    system_message: str = "You are a helpful assistant.",
    model: str = "gpt-4-turbo",
    temperature: float = 0.3,
    json_mode: bool = False,
) -> Union[str, dict]:
    """
        Generate a completion using the OpenAI API.

    Args:
        prompt (str): The user's prompt or query.
        system_message (str, optional): The system message to set the context for the assistant.
            Defaults to "You are a helpful assistant.".
        model (str, optional): The name of the OpenAI model to use for generating the completion.
            Defaults to "gpt-4-turbo".
        temperature (float, optional): The sampling temperature for controlling the randomness of the generated text.
            Defaults to 0.3.
        json_mode (bool, optional): Whether to return the response in JSON format.
            Defaults to False.

    Returns:
        Union[str, dict]: The generated completion.
            If json_mode is True, returns the complete API response as a dictionary.
            If json_mode is False, returns the generated text as a string.
    """

    if json_mode:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            top_p=1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
    else:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            top_p=1,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content


def one_chunk_initial_translation(
    source_lang: str, target_lang: str, source_text: str, tone: int
) -> str:
    """
    Translate the entire text as one chunk using an LLM.
    
    Args:
        source_lang (str): Source language.
        target_lang (str): Target language.
        source_text (str): Text to be translated.
        tone (int): Formality level (1-5).
    
    Returns:
        str: Translated text.
    """

    system_message = f"You are an expert linguist, specializing in translation from {source_lang} to {target_lang}. {tone_mapping[tone]}"

    translation_prompt = f"""Translate the following text from {source_lang} to {target_lang}. \
    Ensure that the translation adheres to the required formality level.

    {source_lang}: {source_text}

    {target_lang}:"""

    translation = get_completion(translation_prompt, system_message=system_message)
    return translation


def one_chunk_reflect_on_translation(
    source_lang: str,
    target_lang: str,
    source_text: str,
    translation_1: str,
    tone: int,
    country: str = "",
) -> str:
    """
    Use an LLM to reflect on the translation, treating the entire text as one chunk.

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language of the translation.
        source_text (str): The original text in the source language.
        translation_1 (str): The initial translation of the source text.
        tone (int): Formality level (1-5).
        country (str): Country specified for the target language.

    Returns:
        str: The LLM's reflection on the translation, providing constructive criticism and suggestions for improvement.
    """

    system_message = f"You are an expert linguist, specializing in translation from {source_lang} to {target_lang}. {tone_mapping[tone]} \
Your goal is to refine the translation to better match the desired tone and clarity."

    if country != "":
        reflection_prompt = f"""Your task is to carefully read a source text and a translation from {source_lang} to {target_lang}, \
and then provide constructive criticism and suggestions to improve the translation. The final style and tone should align with \
the formality level of {tone} and the colloquial usage in {country}.

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation_1}
</TRANSLATION>

Consider the following:
- **Accuracy**: Are there errors of addition, mistranslation, omission, or untranslated text?
- **Fluency**: Does it follow {target_lang} grammar, spelling, and punctuation rules, avoiding unnecessary repetition?
- **Style**: Does it reflect the tone of the source text while considering cultural context?
- **Terminology**: Are technical terms and idioms used correctly and consistently?

Provide a **list of specific, helpful, and constructive suggestions** for improvement.
Output **only** the suggestions and nothing else."""

    else:
        reflection_prompt = f"""Your task is to carefully read a source text and a translation from {source_lang} to {target_lang}, \
and then provide constructive criticism and suggestions to improve the translation. The final style and tone should align with \
the formality level of {tone}.

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation_1}
</TRANSLATION>

Consider the following:
- **Accuracy**: Are there errors of addition, mistranslation, omission, or untranslated text?
- **Fluency**: Does it follow {target_lang} grammar, spelling, and punctuation rules, avoiding unnecessary repetition?
- **Style**: Does it reflect the tone of the source text while considering cultural context?
- **Terminology**: Are technical terms and idioms used correctly and consistently?

Provide a **list of specific, helpful, and constructive suggestions** for improvement.
Output **only** the suggestions and nothing else."""

    reflection = get_completion(reflection_prompt, system_message=system_message)
    return reflection



def one_chunk_improve_translation(
    source_lang: str,
    target_lang: str,
    source_text: str,
    translation_1: str,
    reflection: str,
    tone: int,
) -> str:
    """
    Use the reflection to improve the translation, treating the entire text as one chunk.

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language for the translation.
        source_text (str): The original text in the source language.
        translation_1 (str): The initial translation of the source text.
        reflection (str): Expert suggestions and constructive criticism for improving the translation.
        tone (int): Formality level (1-5).

    Returns:
        str: The improved translation based on the expert suggestions.
    """

    system_message = f"You are an expert linguist, specializing in translation editing from {source_lang} to {target_lang}. {tone_mapping[tone]} \
Your task is to refine the translation while ensuring it maintains the desired tone."

    prompt = f"""Your task is to carefully read, then edit, a translation from {source_lang} to {target_lang}, incorporating expert suggestions and adjusting \
the tone to match the specified formality level ({tone}).

The source text, the initial translation, and the expert linguist suggestions are delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT>, <TRANSLATION></TRANSLATION>, \
and <EXPERT_SUGGESTIONS></EXPERT_SUGGESTIONS> as follows:

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation_1}
</TRANSLATION>

<EXPERT_SUGGESTIONS>
{reflection}
</EXPERT_SUGGESTIONS>

When improving the translation, ensure the following:
**Accuracy**: Correct errors of addition, mistranslation, omission, or untranslated text.  
**Fluency**: Apply {target_lang} grammar, spelling, and punctuation rules while avoiding unnecessary repetitions.  
**Style**: Ensure the translation reflects the original style while adhering to the specified formality level ({tone}).  
**Terminology**: Use domain-specific terminology consistently and select culturally appropriate idioms in {target_lang}.  

**Output only the improved translation and nothing else.**"""

    translation_2 = get_completion(prompt, system_message)
    return translation_2



def one_chunk_translate_text(
    source_lang: str, target_lang: str, source_text: str, tone: int, country: str = ""
) -> str:
    """
    Translate a single chunk of text from the source language to the target language.

    This function performs a two-step translation process:
    1. Get an initial translation of the source text with the specified tone.
    2. Reflect on the initial translation and generate an improved translation.

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language for the translation.
        source_text (str): The text to be translated.
        tone (int): Formality level (1-5).
        country (str): Country specified for the target language.

    Returns:
        str: The improved translation of the source text.
    """
    translation_1 = one_chunk_initial_translation(
        source_lang, target_lang, source_text, tone
    )

    reflection = one_chunk_reflect_on_translation(
        source_lang, target_lang, source_text, translation_1, country
    )

    translation_2 = one_chunk_improve_translation(
        source_lang, target_lang, source_text, translation_1, reflection, tone
    )

    return translation_2



def num_tokens_in_string(
    input_str: str, encoding_name: str = "cl100k_base"
) -> int:
    """
    Calculate the number of tokens in a given string using a specified encoding.

    Args:
        str (str): The input string to be tokenized.
        encoding_name (str, optional): The name of the encoding to use. Defaults to "cl100k_base",
            which is the most commonly used encoder (used by GPT-4).

    Returns:
        int: The number of tokens in the input string.

    Example:
        >>> text = "Hello, how are you?"
        >>> num_tokens = num_tokens_in_string(text)
        >>> print(num_tokens)
        5
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(input_str))
    return num_tokens


def multichunk_initial_translation(
    source_lang: str, target_lang: str, source_text_chunks: List[str], tone: int
) -> List[str]:
    """
    Translate a text in multiple chunks from the source language to the target language.

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language for translation.
        source_text_chunks (List[str]): A list of text chunks to be translated.
        tone (int): Formality level (1-5).

    Returns:
        List[str]: A list of translated text chunks.
    """

    system_message = f"You are an expert linguist, specializing in translation from {source_lang} to {target_lang}. {tone_mapping[tone]}"

    translation_prompt = """Your task is to provide a professional translation from {source_lang} to {target_lang} of PART of a text.
The translation should adhere to the required formality level.

The source text is below, delimited by XML tags <SOURCE_TEXT> and </SOURCE_TEXT>. Translate only the part within the source text
delimited by <TRANSLATE_THIS> and </TRANSLATE_THIS>. You can use the rest of the source text as context, but do not translate any
of the other text. Do not output anything other than the translation of the indicated part of the text.

<SOURCE_TEXT>
{tagged_text}
</SOURCE_TEXT>

To reiterate, you should translate only this part of the text, shown here again between <TRANSLATE_THIS> and </TRANSLATE_THIS>:
<TRANSLATE_THIS>
{chunk_to_translate}
</TRANSLATE_THIS>

Output only the translation of the portion you are asked to translate, and nothing else.
"""

    translation_chunks = []
    for i in range(len(source_text_chunks)):
        # Will translate chunk i
        tagged_text = (
            "".join(source_text_chunks[0:i])
            + "<TRANSLATE_THIS>"
            + source_text_chunks[i]
            + "</TRANSLATE_THIS>"
            + "".join(source_text_chunks[i + 1 :])
        )

        prompt = translation_prompt.format(
            source_lang=source_lang,
            target_lang=target_lang,
            tagged_text=tagged_text,
            chunk_to_translate=source_text_chunks[i],
        )

        translation = get_completion(prompt, system_message=system_message)
        translation_chunks.append(translation)

    return translation_chunks



def multichunk_reflect_on_translation(
    source_lang: str,
    target_lang: str,
    source_text_chunks: List[str],
    translation_1_chunks: List[str],
    tone: int,
    country: str = "",
) -> List[str]:
    """
    Provides constructive criticism and suggestions for improving a partial translation.

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language of the translation.
        source_text_chunks (List[str]): The source text divided into chunks.
        translation_1_chunks (List[str]): The translated chunks corresponding to the source text chunks.
        tone (int): Formality level (1-5).
        country (str): Country specified for the target language.

    Returns:
        List[str]: A list of reflections containing suggestions for improving each translated chunk.
    """

    system_message = f"You are an expert linguist specializing in translation from {source_lang} to {target_lang}. {tone_mapping[tone]} \
You will be provided with a source text and its translation, and your goal is to improve the translation."

    if country != "":
        reflection_prompt = """Your task is to carefully read a source text and part of a translation of that text from {source_lang} to {target_lang}, and then give constructive criticism and helpful suggestions for improving the translation.
The final style and tone of the translation should match the required formality level and the style of {target_lang} colloquially spoken in {country}.

The source text is below, delimited by XML tags <SOURCE_TEXT> and </SOURCE_TEXT>, and the part that has been translated
is delimited by <TRANSLATE_THIS> and </TRANSLATE_THIS> within the source text. You can use the rest of the source text
as context for critiquing the translated part.

<SOURCE_TEXT>
{tagged_text}
</SOURCE_TEXT>

To reiterate, only part of the text is being translated, shown here again between <TRANSLATE_THIS> and </TRANSLATE_THIS>:
<TRANSLATE_THIS>
{chunk_to_translate}
</TRANSLATE_THIS>

The translation of the indicated part, delimited below by <TRANSLATION> and </TRANSLATION>, is as follows:
<TRANSLATION>
{translation_1_chunk}
</TRANSLATION>

When writing suggestions, pay attention to whether there are ways to improve the translation's:\n\
(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),\n\
(ii) fluency (by applying {target_lang} grammar, spelling, and punctuation rules, and ensuring there are no unnecessary repetitions),\n\
(iii) style (by ensuring the translations reflect the style of the source text and match the required formality level),\n\
(iv) terminology (by ensuring terminology use is consistent and reflects the source text domain, and by only ensuring you use equivalent idioms in {target_lang}).\n\

Write a list of specific, helpful, and constructive suggestions for improving the translation.
Each suggestion should address one specific part of the translation.
Output only the suggestions and nothing else."""

    else:
        reflection_prompt = """Your task is to carefully read a source text and part of a translation of that text from {source_lang} to {target_lang}, and then give constructive criticism and helpful suggestions for improving the translation.
The final style and tone of the translation should match the required formality level.

The source text is below, delimited by XML tags <SOURCE_TEXT> and </SOURCE_TEXT>, and the part that has been translated
is delimited by <TRANSLATE_THIS> and </TRANSLATE_THIS> within the source text. You can use the rest of the source text
as context for critiquing the translated part.

<SOURCE_TEXT>
{tagged_text}
</SOURCE_TEXT>

To reiterate, only part of the text is being translated, shown here again between <TRANSLATE_THIS> and </TRANSLATE_THIS>:
<TRANSLATE_THIS>
{chunk_to_translate}
</TRANSLATE_THIS>

The translation of the indicated part, delimited below by <TRANSLATION> and </TRANSLATION>, is as follows:
<TRANSLATION>
{translation_1_chunk}
</TRANSLATION>

When writing suggestions, pay attention to whether there are ways to improve the translation's:\n\
(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),\n\
(ii) fluency (by applying {target_lang} grammar, spelling, and punctuation rules, and ensuring there are no unnecessary repetitions),\n\
(iii) style (by ensuring the translations reflect the style of the source text and match the required formality level),\n\
(iv) terminology (by ensuring terminology use is consistent and reflects the source text domain, and by only ensuring you use equivalent idioms in {target_lang}).\n\

Write a list of specific, helpful, and constructive suggestions for improving the translation.
Each suggestion should address one specific part of the translation.
Output only the suggestions and nothing else."""

    reflection_chunks = []
    for i in range(len(source_text_chunks)):
        # Will reflect on chunk i
        tagged_text = (
            "".join(source_text_chunks[0:i])
            + "<TRANSLATE_THIS>"
            + source_text_chunks[i]
            + "</TRANSLATE_THIS>"
            + "".join(source_text_chunks[i + 1 :])
        )
        if country != "":
            prompt = reflection_prompt.format(
                source_lang=source_lang,
                target_lang=target_lang,
                tagged_text=tagged_text,
                chunk_to_translate=source_text_chunks[i],
                translation_1_chunk=translation_1_chunks[i],
                country=country,
            )
        else:
            prompt = reflection_prompt.format(
                source_lang=source_lang,
                target_lang=target_lang,
                tagged_text=tagged_text,
                chunk_to_translate=source_text_chunks[i],
                translation_1_chunk=translation_1_chunks[i],
            )

        reflection = get_completion(prompt, system_message=system_message)
        reflection_chunks.append(reflection)

    return reflection_chunks


def multichunk_improve_translation(
    source_lang: str,
    target_lang: str,
    source_text_chunks: List[str],
    translation_1_chunks: List[str],
    reflection_chunks: List[str],
    tone: int,
) -> List[str]:
    """
    Improves the translation of a text from source language to target language by considering expert suggestions.

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language for translation.
        source_text_chunks (List[str]): The source text divided into chunks.
        translation_1_chunks (List[str]): The initial translation of each chunk.
        reflection_chunks (List[str]): Expert suggestions for improving each translated chunk.
        tone (int): Formality level (1-5).

    Returns:
        List[str]: The improved translation of each chunk.
    """

    system_message = f"You are an expert linguist, specializing in translation editing from {source_lang} to {target_lang}. {tone_mapping[tone]}"

    improvement_prompt = """Your task is to carefully read, then improve, a translation from {source_lang} to {target_lang}, taking into
account a set of expert suggestions and constructive criticisms. Below, the source text, initial translation, and expert suggestions are provided.

The source text is below, delimited by XML tags <SOURCE_TEXT> and </SOURCE_TEXT>, and the part that has been translated
is delimited by <TRANSLATE_THIS> and </TRANSLATE_THIS> within the source text. You can use the rest of the source text
as context, but need to provide a translation only of the part indicated by <TRANSLATE_THIS> and </TRANSLATE_THIS>.

<SOURCE_TEXT>
{tagged_text}
</SOURCE_TEXT>

To reiterate, only part of the text is being translated, shown here again between <TRANSLATE_THIS> and </TRANSLATE_THIS>:
<TRANSLATE_THIS>
{chunk_to_translate}
</TRANSLATE_THIS>

The translation of the indicated part, delimited below by <TRANSLATION> and </TRANSLATION>, is as follows:
<TRANSLATION>
{translation_1_chunk}
</TRANSLATION>

The expert translations of the indicated part, delimited below by <EXPERT_SUGGESTIONS> and </EXPERT_SUGGESTIONS>, are as follows:
<EXPERT_SUGGESTIONS>
{reflection_chunk}
</EXPERT_SUGGESTIONS>

Taking into account the expert suggestions, rewrite the translation to improve it, ensuring the translation adheres to the required formality level and follows these guidelines:

(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),
(ii) fluency (by applying {target_lang} grammar, spelling, and punctuation rules, and ensuring there are no unnecessary repetitions), \
(iii) style (by ensuring the translations reflect the style of the source text and match the required formality level),
(iv) terminology (inappropriate for context, inconsistent use), or
(v) other errors.

Output only the new translation of the indicated part and nothing else."""

    translation_2_chunks = []
    for i in range(len(source_text_chunks)):
        # Will translate chunk i
        tagged_text = (
            "".join(source_text_chunks[0:i])
            + "<TRANSLATE_THIS>"
            + source_text_chunks[i]
            + "</TRANSLATE_THIS>"
            + "".join(source_text_chunks[i + 1 :])
        )

        prompt = improvement_prompt.format(
            source_lang=source_lang,
            target_lang=target_lang,
            tagged_text=tagged_text,
            chunk_to_translate=source_text_chunks[i],
            translation_1_chunk=translation_1_chunks[i],
            reflection_chunk=reflection_chunks[i],
        )

        translation_2 = get_completion(prompt, system_message=system_message)
        translation_2_chunks.append(translation_2)

    return translation_2_chunks


def multichunk_translation(
    source_lang: str, target_lang: str, source_text_chunks: List[str], tone: int, country: str = ""
) -> List[str]:
    """
    Improves the translation of multiple text chunks based on the initial translation and reflection.

    Args:
        source_lang (str): The source language of the text chunks.
        target_lang (str): The target language for translation.
        source_text_chunks (List[str]): The list of source text chunks to be translated.
        tone (int): Formality level (1-5).
        country (str): Country specified for the target language.

    Returns:
        List[str]: The list of improved translations for each source text chunk.
    """

    translation_1_chunks = multichunk_initial_translation(
        source_lang, target_lang, source_text_chunks, tone
    )

    reflection_chunks = multichunk_reflect_on_translation(
        source_lang,
        target_lang,
        source_text_chunks,
        translation_1_chunks,
        tone,
        country,
    )

    translation_2_chunks = multichunk_improve_translation(
        source_lang,
        target_lang,
        source_text_chunks,
        translation_1_chunks,
        reflection_chunks,
        tone,  # Pass the tone parameter here
    )

    return translation_2_chunks


def calculate_chunk_size(token_count: int, token_limit: int) -> int:
    """
    Calculate the chunk size based on the token count and token limit.

    Args:
        token_count (int): The total number of tokens.
        token_limit (int): The maximum number of tokens allowed per chunk.

    Returns:
        int: The calculated chunk size.

    Description:
        This function calculates the chunk size based on the given token count and token limit.
        If the token count is less than or equal to the token limit, the function returns the token count as the chunk size.
        Otherwise, it calculates the number of chunks needed to accommodate all the tokens within the token limit.
        The chunk size is determined by dividing the token limit by the number of chunks.
        If there are remaining tokens after dividing the token count by the token limit,
        the chunk size is adjusted by adding the remaining tokens divided by the number of chunks.

    Example:
        >>> calculate_chunk_size(1000, 500)
        500
        >>> calculate_chunk_size(1530, 500)
        389
        >>> calculate_chunk_size(2242, 500)
        496
    """

    if token_count <= token_limit:
        return token_count

    num_chunks = (token_count + token_limit - 1) // token_limit
    chunk_size = token_count // num_chunks

    remaining_tokens = token_count % token_limit
    if remaining_tokens > 0:
        chunk_size += remaining_tokens // num_chunks

    return chunk_size


def translate(
    source_lang,
    target_lang,
    source_text,
    tone,
    country,
    max_tokens=MAX_TOKENS_PER_CHUNK,
):
    """Translate the source_text from source_lang to target_lang with a specified tone."""

    num_tokens_in_text = num_tokens_in_string(source_text)

    ic(num_tokens_in_text)

    if num_tokens_in_text < max_tokens:
        ic("Translating text as a single chunk")

        final_translation = one_chunk_translate_text(
            source_lang, target_lang, source_text, tone, country
        )

        return final_translation

    else:
        ic("Translating text as multiple chunks")

        token_size = calculate_chunk_size(
            token_count=num_tokens_in_text, token_limit=max_tokens
        )

        ic(token_size)

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4",
            chunk_size=token_size,
            chunk_overlap=0,
        )

        source_text_chunks = text_splitter.split_text(source_text)

        translation_2_chunks = multichunk_translation(
            source_lang, target_lang, source_text_chunks, tone, country
        )

        return "".join(translation_2_chunks)
