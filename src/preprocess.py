from pathlib import Path
import re, math
import spacy
from tqdm import tqdm
from openai import OpenAI
import textwrap
import tiktoken
from src.config import PREPROCESSING_MODEL, PREPROCESSED_INTERMEDIATE_TOKENS, PREPROCESSED_TARGET_TOKENS

client = OpenAI()
ENCODING = tiktoken.get_encoding("cl100k_base")


class TextPreProcessor:
    """Builds a NetworkX knowledge graph from resolved entities"""
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_md")

        self.BILL_PATTERNS = [
            r"\b\d{2}[-\s]?[A-Z][-]?\d{3,4}\b",  # e.g. 25-O-1271
            r"\bbill\b", r"\bordinance\b", r"\bresolution\b", r"\bmotion\b"
        ]

        self.CONTEXT_KEYWORDS = [
            "approve","approved","pass","vote","rejected","held","amendment",
            "funding","budget","project","development","zoning","property",
            "contract","department","finance","council","committee"
        ]

        self.NOISE_PATTERNS = [
            r"\b(seconded?|moved?)\b",
            r"\bvote is (open|closed)\b",
            r"\b(all|everyone) (in favor|against)\b",
            r"\bprint that screen\b",
            r"\bthank you\b",
            r"\b(good (afternoon|morning|evening))\b",
            r"\bplease (take your seats|come forward)\b",
            r"\bany discussion\b",
            r"\b(public comment|hearing)\b"
        ]


    def clean_text(self, text: str) -> str:
        """Remove speaker labels, procedural fragments, and noise patterns."""
        
        text = re.sub(r"^[A-Z][A-Z\s\.\-']{2,20}:\s*", "", text, flags=re.M)        
        parts = re.split(r'(?<=[.!?])\s+', text)
        
        cleaned = []
        for line in parts:
            line = line.strip()
            if len(line.split()) < 3:
                continue

            if any(re.search(p, line, re.I) for p in self.NOISE_PATTERNS):
                continue

            cleaned.append(line)
        
        return " ".join(cleaned)


    def bill_signal(self, sent: str) -> bool:
        return any(re.search(p, sent, re.I) for p in self.BILL_PATTERNS)


    def score_sentence(self, sent: str) -> float:
        """
        Score each sentence between 0-100.
        Bill-related sentences instantly get 100 and are top priority.
        """
        if self.bill_signal(sent):
            return 100.0

        doc = self.nlp(sent)
        score = 0.0

        for ent in doc.ents:
            if ent.label_ in {"LAW", "ORG", "MONEY", "GPE"}:
                score += 10

        for w in self.CONTEXT_KEYWORDS:
            if w in sent.lower():
                score += 5

        if any(t.pos_ == "VERB" for t in doc):
            score += 2

        return min(score, 99.0)


    def reduce_transcript(self, text: str, keep_ratio: float = 0.10) -> str:

        text = self.clean_text(text)
        doc = self.nlp(text)
        sents = [s.text.strip() for s in doc.sents if s.text.strip()]
        scored = [(s, self.score_sentence(s)) for s in sents]

        k = max(1, math.ceil(len(scored) * keep_ratio))
        top_sents = sorted(scored, key=lambda x: x[1], reverse=True)[:k]

        top_sents.sort(key=lambda x: sents.index(x[0]))
        top_texts = [s for s, _ in top_sents]

        grouped = []
        buffer = []
        last_idx = None
        for s in top_texts:
            idx = sents.index(s)
            if last_idx is None or idx - last_idx <= 2:
                buffer.append(s)
            else:
                grouped.append(" ".join(buffer))
                buffer = [s]
            last_idx = idx
        
        if buffer:
            grouped.append(" ".join(buffer))

        return "\n\n".join(grouped)


    def summarize_text(self, text: str) -> str:
        """
        Compress a reduced transcript (~20k tokens) into a focused 2k-token summary.
        Keeps all legislative details about Bills / Ordinances / Resolutions.
        """

        limited_tokens = ENCODING.encode(f'### TRANSCRIPT INPUT: \n {text}')
        final_reduced_text = ENCODING.decode(limited_tokens[:PREPROCESSED_INTERMEDIATE_TOKENS])

        prompt = textwrap.dedent(f"""
            You are a legislative summarization assistant.

            TASK:
                Create a detailed summary not more than {PREPROCESSED_TARGET_TOKENS} tokens of the city-council transcript below. Another LLM will extract structured entities from this summary.
                The summary should be factual, focuses on legislative actions and covers all the important points/features considerder.

            KEY FOCUS AREAS:
                • Bills / Ordinances / Resolutions and their identifiers
                • Outcomes (approved, rejected, tabled)
                • Key participants and departments
                • Any Projects or funding actions

            REQUIREMENTS:
                - Preserve every bill ID and its decision.
                - Keep chronology and structure clear.
                - Exclude small talk, procedural chatter, and greetings.
                - Use concise factual language (no speculation).
                - Output plain text paragraphs.
                - Ensure all legislative details are included.
                - Target length: {PREPROCESSED_TARGET_TOKENS} tokens.
            
            {final_reduced_text}   
        """)

        response = client.chat.completions.create(
            model=PREPROCESSING_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=PREPROCESSED_TARGET_TOKENS,
        )

        return response.choices[0].message.content.strip()
    

    def get_preprocessed_summary(self, transcript) -> str:
        reduced_transcript = self.reduce_transcript(transcript, keep_ratio=0.10)
        summarised_text = self.summarize_text(reduced_transcript)

        return summarised_text
    
