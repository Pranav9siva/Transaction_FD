import networkx as nx
import pandas as pd
from collections import Counter

class FraudGraphDetector:
    """
    Graph-Based Fraud Detection module using NetworkX multigraph as described in Section IV-F.
    - Hub Detection (nodes degree > threshold, default 10)
    - Subgraph Analysis (weakly connected components >= 3)
    - Cycle Detection (simple cycles A -> B -> C -> A)
    - PageRank Risk Scoring (alpha = 0.85)
    """
    def __init__(self, df=None, sender_col='card1', receiver_col='card2'):
        self.sender_col = sender_col
        self.receiver_col = receiver_col
        self.graph = nx.MultiDiGraph()
        
        if df is not None and not df.empty:
            self.build_graph(df)

    def build_graph(self, df):
        """Constructs/updates the directed multigraph from transaction rows."""
        edges = []
        for idx, row in df.iterrows():
            sender = row.get(self.sender_col)
            receiver = row.get(self.receiver_col)
            
            if pd.notna(sender) and pd.notna(receiver):
                attr = {'transaction_id': str(row.get('TransactionID', idx))}
                if 'TransactionAmt' in row:
                    attr['amount'] = float(row['TransactionAmt'])
                if 'isFraud' in row:
                    attr['isFraud'] = int(row['isFraud'])
                edges.append((sender, receiver, attr))
                
        self.graph.add_edges_from(edges)

    def add_transaction(self, sender, receiver, amount=0.0, transaction_id=None):
        """Dynamically add an edge in streaming mode."""
        if sender and receiver:
            self.graph.add_edge(sender, receiver, amount=amount, transaction_id=transaction_id)

    def detect_highly_connected_nodes(self, threshold=10):
        """Hub Detection: nodes with degree > threshold (default 10)."""
        degree_dict = dict(self.graph.degree())
        return {str(node): deg for node, deg in degree_dict.items() if deg > threshold}

    def detect_suspicious_subgraphs(self, min_component_size=3):
        """Subgraph Analysis: weakly connected components size >= 3."""
        if self.graph.number_of_nodes() == 0:
            return []
        components = list(nx.weakly_connected_components(self.graph))
        suspicious = [[str(node) for node in c] for c in components if len(c) >= min_component_size]
        suspicious.sort(key=len, reverse=True)
        return suspicious

    def get_cycles(self):
        """Cycle Detection: simple cycles A -> B -> C -> A."""
        if self.graph.number_of_nodes() == 0 or self.graph.number_of_nodes() > 1000:
            return []
        try:
            cycles = list(nx.simple_cycles(self.graph))
            return [[str(n) for n in cycle] for cycle in cycles]
        except Exception as e:
            print(f"[GraphDetector] Cycle detection skipped/error: {e}")
            return []

    def calculate_pagerank_risk(self, alpha=0.85):
        """PageRank Risk Scoring (alpha = 0.85)."""
        if self.graph.number_of_nodes() == 0:
            return {}
        try:
            scores = nx.pagerank(self.graph, alpha=alpha)
            return {str(node): float(score) for node, score in scores.items()}
        except Exception:
            deg = dict(self.graph.degree())
            max_deg = max(deg.values()) if deg else 1
            return {str(node): float(d / max_deg) for node, d in deg.items()}

    def get_node_risk(self, node_id, alpha=0.85):
        """Calculates normalized graph risk score for a given node (0.0 to 1.0)."""
        if not node_id or not self.graph.has_node(node_id):
            return 0.0
        
        pr_scores = self.calculate_pagerank_risk(alpha=alpha)
        pr_val = pr_scores.get(str(node_id), 0.0)
        
        # Combine PageRank with degree score
        degree = self.graph.degree(node_id)
        deg_risk = min(degree / 20.0, 1.0)
        
        graph_risk = min((pr_val * 5.0) + (deg_risk * 0.5), 1.0)
        return float(graph_risk)

if __name__ == "__main__":
    detector = FraudGraphDetector()
    detector.add_transaction("Card_A", "Merch_B", 100.0)
    detector.add_transaction("Merch_B", "Card_C", 100.0)
    detector.add_transaction("Card_C", "Card_A", 100.0)
    print("Cycles:", detector.get_cycles())
    print("Card_A Risk:", detector.get_node_risk("Card_A"))
