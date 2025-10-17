import sys
import time
from pathlib import Path

from src.extract import EntityExtractor
from src.resolve import EntityResolver
from src.graph import KnowledgeGraphBuilder


class Pipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self):
        self.start_time = None
    
    def log_step(self, step_num: int, step_name: str):
        """Log pipeline step"""
        print("\n" + "="*70)
        print(f"STEP {step_num}: {step_name}")
        print("="*70 + "\n")
    
    def run(self):
        """Run the complete pipeline"""
        self.start_time = time.time()
        
        print("\n" + "🚀" * 35)
        print("MUNICIPAL KNOWLEDGE GRAPH PIPELINE")
        print("🚀" * 35)
        
        # Step 1: Entity Extraction
        self.log_step(1, "ENTITY EXTRACTION & PREDICTION")
        extractor = EntityExtractor()
        extractions = extractor.extract_from_directory()
        
        if not extractions:
            print("No extractions found. Exiting.")
            sys.exit(1)
        
        # Step 2: Entity Resolution
        self.log_step(2, "ENTITY RESOLUTION & DEDUPLICATION")
        resolver = EntityResolver()
        aggregated = resolver.aggregate_entities()
        resolution = resolver.resolve_entities(aggregated)
        resolver.save_resolution(resolution)
        
        # Step 3: Knowledge Graph Construction
        self.log_step(3, "KNOWLEDGE GRAPH CONSTRUCTION")
        builder = KnowledgeGraphBuilder()
        builder.build_graph()
        builder.export_graph()
        
        # Success
        elapsed = time.time() - self.start_time
        print("\n" + "✅" * 35)
        print(f"PIPELINE COMPLETED SUCCESSFULLY")
        print(f"Total time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
        print("✅" * 35 + "\n")
        


def main():
    """Main entry point"""
    pipeline = Pipeline()
    pipeline.run()


if __name__ == "__main__":
    main()