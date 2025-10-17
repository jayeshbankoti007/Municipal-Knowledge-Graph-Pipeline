import json
import re
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
from difflib import SequenceMatcher

from src.config import EXTRACTIONS_DIR, OUTPUT_DIR
from src.models import TranscriptExtraction


class EntityResolver:
    """Pure rule-based entity resolver producing alias→canonical dicts."""

    def __init__(self):
        self.common_abbreviations = {
            'dept': 'department',
            'div': 'division',
            'comm': 'committee',
            'atl': 'atlanta',
            'dev': 'development',
            'mgmt': 'management',
            'fin': 'finance',
            'hr': 'human resources',
            'it': 'information technology',
            'apd': 'atlanta police department',
            'afd': 'atlanta fire department',
            'dot': 'department of transportation',
        }

    @staticmethod
    def normalize_bill_id(bill_id: str) -> str:
        """Normalize bill IDs to standard format."""
        bill_id = re.sub(r'^(bill|ordinance|resolution)\s*', '', bill_id, flags=re.IGNORECASE)
        bill_id = bill_id.upper().strip()
        match = re.match(r'(\d{2})-?([OR])-?(\d+)', bill_id)
        if match:
            return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
        return bill_id

    def normalize_text(self, text: str) -> str:
        """Normalize text for fuzzy matching."""
        text = text.lower().strip()
        words = text.split()
        return ' '.join(self.common_abbreviations.get(word, word) for word in words)

    def fuzzy_match_score(self, s1: str, s2: str) -> float:
        """Calculate fuzzy matching score between two strings."""
        s1_norm, s2_norm = self.normalize_text(s1), self.normalize_text(s2)
        if s1_norm == s2_norm:
            return 1.0
        if s1_norm in s2_norm or s2_norm in s1_norm:
            return 0.85
        return SequenceMatcher(None, s1_norm, s2_norm).ratio()

    def resolve_bills(self, bills: List[str]) -> Dict[str, str]:
        """Resolve bills into alias→canonical dict."""
        bill_map = defaultdict(list)
        for bill in bills:
            normalized = self.normalize_bill_id(bill)
            bill_map[normalized].append(bill)

        lookup = {}
        for canonical, all_forms in bill_map.items():
            lookup[canonical] = canonical
            for form in all_forms:
                lookup[form] = canonical
        return lookup

    def resolve_fuzzy(self, entities: List[str], threshold: float = 0.85) -> Dict[str, str]:
        """Resolve organizations/projects with fuzzy matching into alias→canonical dict."""
        processed = set()
        entity_list = sorted(set(entities))
        lookup = {}

        for i, entity1 in enumerate(entity_list):
            if entity1 in processed:
                continue

            lookup[entity1] = entity1
            processed.add(entity1)

            for j, entity2 in enumerate(entity_list):
                if i != j and entity2 not in processed:
                    if self.fuzzy_match_score(entity1, entity2) >= threshold:
                        lookup[entity2] = entity1
                        processed.add(entity2)

        return lookup

    def aggregate_entities(self) -> Dict[str, List[str]]:
        """Load and aggregate all extracted entities."""
        aggregated = defaultdict(list)
        extraction_files = list(EXTRACTIONS_DIR.glob("*.json"))
        print(f"📂 Loading {len(extraction_files)} extraction files...")

        for file_path in extraction_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    extraction = TranscriptExtraction(**data)
                    aggregated['organizations'].extend([o.name for o in extraction.organizations])
                    aggregated['bills'].extend([b.id for b in extraction.bills])
                    aggregated['projects'].extend([p.name for p in extraction.projects])
            except Exception as e:
                print(f"⚠️  Error loading {file_path.name}: {e}")

        # Remove exact duplicates
        for key in aggregated:
            aggregated[key] = sorted(set(aggregated[key]))

        return aggregated

    def resolve_entities(self, aggregated: Dict[str, List[str]]) -> Dict[str, Dict[str, str]]:
        """Resolve all entities into alias→canonical dicts."""
        print("🔧 Resolving bills...")
        bills_lookup = self.resolve_bills(aggregated['bills'])
        print(f"  {len(aggregated['bills'])} bills → {len(set(bills_lookup.values()))} unique")

        print("🔧 Resolving organizations...")
        orgs_lookup = self.resolve_fuzzy(aggregated['organizations'])
        print(f"  {len(aggregated['organizations'])} orgs → {len(set(orgs_lookup.values()))} unique")

        print("🔧 Resolving projects...")
        projects_lookup = self.resolve_fuzzy(aggregated['projects'])
        print(f"  {len(aggregated['projects'])} projects → {len(set(projects_lookup.values()))} unique")

        return {
            "bills": bills_lookup,
            "organizations": orgs_lookup,
            "projects": projects_lookup,
        }

    def save_resolution(self, resolved_dict: Dict[str, Dict[str, str]]):
        output_file = OUTPUT_DIR / "resolved_entities_dict.json"
        with open(output_file, 'w') as f:
            json.dump(resolved_dict, f, indent=2)
        print(f"\n✅ Saved resolved entities to {output_file}")


def main():
    resolver = EntityResolver()
    aggregated = resolver.aggregate_entities()
    resolved_dict = resolver.resolve_entities(aggregated)
    resolver.save_resolution(resolved_dict)
    print("\n✅ Resolution complete!")



if __name__ == "__main__":
    main()