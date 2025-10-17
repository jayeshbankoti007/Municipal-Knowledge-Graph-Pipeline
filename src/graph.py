import json
import pickle
from pathlib import Path
from typing import Dict, List
import networkx as nx
from collections import Counter
import matplotlib.pyplot as plt
from enum import Enum

from src.config import EXTRACTIONS_DIR, OUTPUT_DIR, KG_FILE_PATH, RESOLUTION_FILE, KG_NEO4J_PATH
from src.models import TranscriptExtraction
from src.visualisation import visualize_knowledge_graph


class KnowledgeGraphBuilder:
    """Builds a NetworkX knowledge graph from resolved entities"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.stats = {
            'nodes': Counter(),
            'edges': Counter()
        }
        self.alias_map = {}


    def load_resolutions(self) -> Dict[str, Dict[str, str]]:
        """Load resolved entities as alias→canonical dicts"""

        if not RESOLUTION_FILE.exists():
            raise FileNotFoundError(f"Resolution file not found: {RESOLUTION_FILE}")
        
        with open(RESOLUTION_FILE, 'r') as f:
            self.alias_map = json.load(f)


    def load_extractions(self) -> List[TranscriptExtraction]:
        """Load all extraction files"""
        extractions = []
        
        for file_path in EXTRACTIONS_DIR.glob("*.json"):
            with open(file_path, 'r') as f:
                data = json.load(f)
                extractions.append(TranscriptExtraction(**data))
        
        return extractions


    def resolve_name(self, name: str) -> str:
        """Resolve a name to its canonical form"""
        return self.alias_map.get(name.lower(), name)


    def add_nodes(self, extractions: List[TranscriptExtraction]):
        """Add all nodes to the graph"""

        for extraction in extractions:
            for Person in extraction.people:
                person_node = f"person:{Person.name}"                
                self.graph.add_node(
                    person_node,
                    type='Person',
                    name=Person.name,
                    role=Person.role if Person.role else "member",
                    organization=Person.organization if Person.organization else "City Council"
                )
                self.stats['nodes']['Person'] += 1

            for Organization in extraction.organizations:
                org_name = self.resolve_name(Organization.name)
                ord_node = f"org:{org_name}"                
                self.graph.add_node(
                    ord_node,
                    name = org_name,
                    type = 'Organization',
                    org_type = Organization.type if Organization.type else 'Missing',   
                )
                self.stats['nodes']['Organization'] += 1

            for Bill in extraction.bills:
                bill_name = self.resolve_name(Bill.id)
                bill_node = f"bill:{bill_name}"                
                self.graph.add_node(
                    bill_node,
                    type='Bill',
                    title=Bill.title,
                    bill_type=Bill.type if Bill.type else "Missing",
                    prediction=Bill.prediction if Bill.prediction else "Missing",
                    confidence=Bill.confidence if Bill.confidence else "Missing",
                    reasoning=Bill.reasoning if Bill.reasoning else "Missing",
                )
                self.stats['nodes']['Bill'] += 1

            for Project in extraction.projects:
                project_name = self.resolve_name(Project.name)
                project_node = f"project:{project_name}"
                self.graph.add_node(
                    project_node,
                    name=Project.name,
                    type='Project',
                    project_type=Project.type if Project.type else "Missing",
                    location=Project.location if Project.location else "Unknown",
                    amount=Project.amount if Project.amount else "Unknown",
                )
                self.stats['nodes']['Project'] += 1


    def add_edges(self, extractions: List[TranscriptExtraction]):
        """Add all edges/relationships to the graph"""

        for extraction in extractions:
            
            # Add VOTED edges
            for Vote in extraction.votes:
                bill_id = self.resolve_name(Vote.bill_id)
                bill_node = f"bill:{bill_id}"
                person_node = f"person:{Vote.person}"

                if bill_node in self.graph and person_node in self.graph:
                    self.graph.add_edge(
                        person_node,
                        bill_node,
                        relation='VOTED_ON',
                        vote=Vote.vote
                    )
                    self.stats['edges']['VOTED_ON'] += 1
            

            # Add MEMBER_OF edges (Person → Organization)
            for person in extraction.people:
                if person.organization:
                    org_name = self.resolve_name(person.organization)
                    person_node = f"person:{person.name}"
                    org_node = f"org:{org_name}"
                    
                    if person_node in self.graph and org_node in self.graph:
                        self.graph.add_edge(
                            person_node,
                            org_node,
                            relation='MEMBER_OF',
                            role=person.role
                        )
                        self.stats['edges']['MEMBER_OF'] += 1


            # Add MENTIONED_IN edges (Person → Bill)
            if extraction.people and extraction.bills:
                for person in extraction.people:
                    person_node = f"person:{person.name}"
                    for bill in extraction.bills:
                        bill_id = self.resolve_name(bill.id)
                        bill_node = f"bill:{bill_id}"
                        
                        if person_node in self.graph and bill_node in self.graph:
                            # Only add if not already connected by VOTED_ON
                            if not self.graph.has_edge(person_node, bill_node):
                                self.graph.add_edge(
                                    person_node,
                                    bill_node,
                                    relation='MENTIONED_IN'
                                )
                                self.stats['edges']['MENTIONED_IN'] += 1


            # Add AUTHORIZES edges (Bill → Project)
            for Project in extraction.projects:
                project_name = self.resolve_name(Project.name)
                project_node = f"project:{project_name}"

                for bill in extraction.bills:
                    bill_id = self.resolve_name(bill.id)
                    bill_node = f"bill:{bill_id}"
                    
                    # Check if project mentioned in bill title
                    if (bill_node in self.graph and project_node in self.graph 
                        and Project.name.lower() in bill.title.lower()):
                        self.graph.add_edge(
                            bill_node,
                            project_node,
                            relation='AUTHORIZES'
                        )
                        self.stats['edges']['AUTHORIZES'] += 1


            # Add RELATES_TO edges (Bill → Organization)
            for Bill in extraction.bills:
                bill_id = self.resolve_name(Bill.id)
                bill_node = f"bill:{bill_id}"
                
                for Organization in extraction.organizations:
                    org_name = self.resolve_name(Organization.name)
                    org_node = f"org:{org_name}"
                    
                    if (bill_node in self.graph and org_node in self.graph):
                        self.graph.add_edge(
                            bill_node,
                            org_node,
                            relation='RELATES_TO'
                        )
                        self.stats['edges']['RELATES_TO'] += 1


    def build_graph(self):
        """Main method to build the complete knowledge graph"""
        print("\n" + "="*60)
        print("BUILDING KNOWLEDGE GRAPH")
        print("="*60)
        
        # Load data
        print("\n1. Loading resolved entities and Creating alias map...")
        self.load_resolutions()
        
        print("2. Loading extractions...")
        extractions = self.load_extractions()
        
        print("3. Adding nodes...")
        self.add_nodes(extractions)
        
        print("4. Adding edges...")
        self.add_edges(extractions)
        
        print("5. Graph visualisation...")
        try:
            self.export_graph()
            self.export_GraphML()
            visualize_knowledge_graph()
        except Exception as e:
            print(f"⚠️  Visualization failed: {e}")

        # Print stats
        print("\n" + "="*60)
        print("GRAPH STATISTICS")
        print("="*60)
        print(f"Total Nodes: {self.graph.number_of_nodes()}")
        for node_type, count in self.stats['nodes'].items():
            print(f"  {node_type}: {count}")
        
        print(f"\nTotal Edges: {self.graph.number_of_edges()}")
        
        for edge_type, count in self.stats['edges'].items():
            print(f"  {edge_type}: {count}")

        print("="*60 + "\n")


    def export_graph(self):
        """Export graph in multiple formats"""
        with open(KG_FILE_PATH, 'wb') as f:
            pickle.dump(self.graph, f)
        print(f"✅ Saved NetworkX graph: {KG_FILE_PATH}")


    def export_GraphML(self):
        """Export graph in multiple formats"""
        for node, attrs in self.graph.nodes(data=True):
            for key, val in attrs.items():
                if isinstance(val, Enum):
                    attrs[key] = val.name
        
        for u, v, attrs in self.graph.edges(data=True):
            for key, val in attrs.items():
                if isinstance(val, Enum):
                    attrs[key] = val.name
        
        nx.write_graphml(self.graph, KG_NEO4J_PATH)
        print(f"✅ Saved GraphML: {KG_NEO4J_PATH}")
                

def main():
    """Main graph construction function"""
    builder = KnowledgeGraphBuilder()
    builder.build_graph()
    builder.export_graph()
    builder.export_GraphML()
    visualize_knowledge_graph()

if __name__ == "__main__":
    main()