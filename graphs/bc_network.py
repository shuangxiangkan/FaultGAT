import networkx as nx
from .base_graph import BaseKRegularGraph

class BCNetwork(BaseKRegularGraph):
    """BC Network implementation class"""
    
    def _get_k_value(self):
        """BC Network's regularity k equals parameter n"""
        return self.n
    
    def get_graph_type(self):
        """Return graph type name"""
        return "bc"
    
    def _calculate_theoretical_diagnosability(self):
        """
        Calculate theoretical intermittent fault diagnosability of BC Network.
        According to table data: BC Networks (X_n): t_i = n - 3 (when n≥4)
        """
        return self.n - 3
    
    def _generate_graph(self):
        """Generate BC Network (using hypercube as BC Network representative)."""
        # For BC Network, n represents dimension, node count is 2^n, each node has n neighbors
        # Here we use hypercube as BC Network representative
        return nx.hypercube_graph(self.n) 