import json
from pathlib import Path
from typing import List
import pandas as pd
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from src.config import OPENAI_API_KEY, MODEL_NAME, DATA_DIR, EXTRACTIONS_DIR, TEMPERATURE
from src.models import TranscriptExtraction
from src.preprocess import TextPreProcessor


class EntityExtractor:
    """Extracts entities and makes predictions from transcripts"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=MODEL_NAME,
            api_key=OPENAI_API_KEY,
            temperature=TEMPERATURE
        )
        self.parser = PydanticOutputParser(pydantic_object=TranscriptExtraction)
        self.prompt = self._create_prompt()
        self.metadata_df = None
        self.preprocess = TextPreProcessor()


    def load_metadata(self, csv_path: Path = None):
        """Load metadata CSV to map files to meeting info"""
        if csv_path is None:
            csv_path = DATA_DIR.parent / "Atlanta_meetings.csv"
        
        if csv_path.exists():
            self.metadata_df = pd.read_csv(csv_path)
            print(f"Loaded metadata for {len(self.metadata_df)} meetings")
        else:
            print(f"Warning: Metadata CSV not found at {csv_path}")


    def _create_prompt(self) -> ChatPromptTemplate:
        """Create extraction prompt template"""
        template = """You are an expert at extracting structured information from the summarised city council transcripts.
            You will be given a summary of a city council meeting transcript.

            MEETING INFO:
            Date: {meeting_date}
            Title: {meeting_title}

            Extract the following entities from the summary of the transcript:

            1. **Bills/Legislation**: Any ordinances, resolutions, or legislation
            - Extract the ID (e.g., "25-O-1271", "25-R-3450")
            - Title/description
            
            2. **Prediction for each Bill**:
            - Status: APPROVED / REJECTED / UNCERTAIN
            - Confidence: HIGH / MEDIUM / LOW
            - Reasoning based on:
                * Was it approved/rejected in the transcript?
                * Voting patterns (unanimous, split, held)
                * Discussion sentiment (concerns, support)
                * Amendments or holds mentioned
            
            3. **People**: Council members, speakers, officials
            - Full name
            - Role/title if mentioned else "member"
            - Organization affiliation if mentioned else "City Council"
            
            4. **Organizations**: Departments, companies, agencies
            - Full name (e.g., "Department of Finance", "HR", "Public Works")
            - Type (department, company, agency)
            
            5. **Projects** (Real Estate/Infrastructure):
            - Project name or description
            - Type (residential, commercial, infrastructure)
            - Location/address if mentioned
            - Dollar amount if mentioned
            
            6. **Votes**: Explicit votes on bills
            - Bill ID
            - Person name
            - Vote (yes/no/held/abstain)

            TRANSCRIPT:
                {transcript}
                {format_instructions}

            IMPORTANT GUIDELINES:
                - Bill IDs: Normalize format (e.g., "25-O-1271" not "25-o-1271" or "Ordinance 25-O-1271")
                - People: Use full formal names consistently (e.g., "Howard Shook" not "Mr. Shook" or "Chairman Shook")
                - Organizations: Use full official names (e.g., "Department of Finance" not "DOF" or "Finance")
                - Predictions:
                    * If bill explicitly approved/passed → APPROVED (HIGH confidence)
                    * If bill held/tabled → UNCERTAIN (MEDIUM confidence)
                    * If significant opposition mentioned → REJECTED or UNCERTAIN (LOW-MEDIUM confidence)
                    * Look for phrases like "vote is closed", "motion passes", "hold", "substitute"

            """
        return ChatPromptTemplate.from_messages([
            ("system", "You are a precise entity extraction assistant specializing in municipal government transcripts."),
            ("human", template)
        ])


    def load_transcript(self, file_path: Path) -> tuple[str, dict]:
        """Load transcript from JSON file and extract text"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract text from segments
        if isinstance(data, list):
            transcript_text = " ".join([
                item.get('text', '') 
                for item in data 
                if isinstance(item, dict) and 'text' in item
            ])
        else:
            transcript_text = str(data)
        
        # Get metadata if available
        metadata = {}
        if self.metadata_df is not None:
            filename = file_path.name
            match = self.metadata_df[self.metadata_df['s3_uri'].str.contains(filename)]
            if not match.empty:
                metadata = {
                    'date': match.iloc[0]['runlink_date'],
                    'title': match.iloc[0]['runlink_title'],
                    'url': match.iloc[0]['runlink_url']
                }
        
        return transcript_text, metadata


    def extract_from_file(self, file_path: Path) -> TranscriptExtraction:
        """Extract entities from a single transcript file"""

        transcript_text, metadata = self.load_transcript(file_path)
        reduced_transcript_text = self.preprocess.get_preprocessed_summary(transcript_text)
        print(f"  Extracted transcript {len(transcript_text)} -> reduced to {len(reduced_transcript_text)} tokens")
        
        chain = self.prompt | self.llm | self.parser        

        result = chain.invoke({
            "transcript": reduced_transcript_text,
            "meeting_date": metadata.get('date', 'Unknown'),
            "meeting_title": metadata.get('title', 'Unknown'),
            "format_instructions": self.parser.get_format_instructions()
        })
        
        # Save extraction
        output_file = EXTRACTIONS_DIR / f"{file_path.stem}.json"
        output_data = result.dict()
        output_data['metadata'] = metadata
        output_data['source_file'] = file_path.name
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        return result
            
    
    def extract_from_directory(self, directory: Path = DATA_DIR) -> List[TranscriptExtraction]:
        """Extract entities from all transcript JSON files"""
        # Load metadata first
        self.load_metadata()
        
        # Find all JSON transcript files
        transcript_files = sorted(directory.glob("**/*.json"))
        
        # Filter out metadata.csv related files
        transcript_files = [f for f in transcript_files if 'metadata' not in f.name.lower()]
        
        print(f"\n{'='*60}")
        print(f"Found {len(transcript_files)} transcript files")
        print(f"{'='*60}\n")
        
        extractions = []
        for file_path in tqdm(transcript_files, desc="Extracting entities", unit="file"):
            extraction = self.extract_from_file(file_path)
            extractions.append(extraction)
        
        print(f"\n{'='*60}")
        print(f"✅ Successfully processed {len(extractions)} transcripts")
        print(f"{'='*60}\n")
        
        return extractions

def main():
    """Main extraction function"""
    extractor = EntityExtractor()
    extractions = extractor.extract_from_directory()
    
    # Calculate statistics
    summary = {
        "total_transcripts": len(extractions),
        "total_bills": sum(len(e.bills) for e in extractions),
        "total_people": sum(len(e.people) for e in extractions),
        "total_organizations": sum(len(e.organizations) for e in extractions),
        "total_projects": sum(len(e.projects) for e in extractions),
        "total_votes": sum(len(e.votes) for e in extractions),
        "predictions": {
            "approved": sum(1 for e in extractions for b in e.bills if b.prediction == "APPROVED"),
            "rejected": sum(1 for e in extractions for b in e.bills if b.prediction == "REJECTED"),
            "uncertain": sum(1 for e in extractions for b in e.bills if b.prediction == "UNCERTAIN")
        }
    }

    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    print(f"Total Transcripts:    {summary['total_transcripts']}")
    print(f"Total Bills:          {summary['total_bills']}")
    print(f"Total People:         {summary['total_people']}")
    print(f"Total Organizations:  {summary['total_organizations']}")
    print(f"Total Projects:       {summary['total_projects']}")
    print(f"Total Votes:          {summary['total_votes']}")
    print(f"\nPredictions:")
    print(f"  Approved:    {summary['predictions']['approved']}")
    print(f"  Rejected:    {summary['predictions']['rejected']}")
    print(f"  Uncertain:   {summary['predictions']['uncertain']}")
    print("="*60 + "\n")
    
    # Save summary
    with open(EXTRACTIONS_DIR.parent / "extraction_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"💾 Summary saved to: {EXTRACTIONS_DIR.parent / 'extraction_summary.json'}")

if __name__ == "__main__":
    main()