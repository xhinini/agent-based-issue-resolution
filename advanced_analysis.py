import ast
import networkx as nx
from typing import Tuple, Set, Dict, Any

# ================================
# CFG Building
# ================================

class CFGBuilder(ast.NodeVisitor):
    """
    Builds a Control Flow Graph (CFG) for a Python function.
    
    The CFG is represented as a networkx.DiGraph where:
      - Each node is an integer (a unique node id).
      - Node attributes:
          * 'label': a string label (e.g., "entry", "if (<cond>)", "merge", etc.)
          * 'stmts': a list of string representations of statements (from ast.dump)
    
      - Sequential statements (appending them to the current basic block)
      - if/else branches (splitting and merging control flow)
      - while loops and for loops (creating loop condition, body, and exit nodes)
    """
    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_counter = 0
        # Create a unique entry node and initialize current_nodes list
        self.entry_node = self._new_node("entry")
        self.current_nodes = [self.entry_node]

    def _new_node(self, label: str, stmts=None) -> int:
        """Create a new CFG node with a given label and optional list of statements."""
        node_id = self.node_counter
        self.node_counter += 1
        if stmts is None:
            stmts = []
        self.graph.add_node(node_id, label=label, stmts=stmts)
        return node_id

    def add_edge(self, src: int, dst: int):
        self.graph.add_edge(src, dst)

    def build(self, node: ast.AST) -> nx.DiGraph:
        """
        Build a CFG from an AST node. For a function, call this on the FunctionDef node.
        """
        self.visit(node)
        # Connect any leftover current nodes to an explicit exit node.
        exit_node = self._new_node("exit")
        for cn in self.current_nodes:
            self.add_edge(cn, exit_node)
        return self.graph

    def generic_visit(self, node):
        """
        For any statement that is not explicitly handled, add its ast.dump representation
        to each current basic block.
        """
        if isinstance(node, ast.stmt):
            for cn in self.current_nodes:
                stmts = self.graph.nodes[cn].get("stmts", [])
                stmts.append(ast.dump(node))
                self.graph.nodes[cn]["stmts"] = stmts
        super().generic_visit(node)

    def visit_If(self, node: ast.If):
        # Create a condition node
        cond_node = self._new_node("if (" + ast.unparse(node.test) + ")")
        for cn in self.current_nodes:
            self.add_edge(cn, cond_node)

        # Save the current nodes for later merging.
        prev_nodes = self.current_nodes.copy()

        # Then branch:
        self.current_nodes = [self._new_node("then")]
        for stmt in node.body:
            self.visit(stmt)
        then_exit_nodes = self.current_nodes.copy()

        # Else branch:
        if node.orelse:
            self.current_nodes = [self._new_node("else")]
            for stmt in node.orelse:
                self.visit(stmt)
            else_exit_nodes = self.current_nodes.copy()
        else:
            # If no else, the false branch is the condition node itself.
            else_exit_nodes = [cond_node]

        # Merge branch: create a merge node and link all branch exits.
        merge_node = self._new_node("merge")
        for n in then_exit_nodes + else_exit_nodes:
            self.add_edge(n, merge_node)
        self.current_nodes = [merge_node]

    def visit_While(self, node: ast.While):
        loop_cond = self._new_node("while (" + ast.unparse(node.test) + ")")
        for cn in self.current_nodes:
            self.add_edge(cn, loop_cond)
        # Loop body:
        self.current_nodes = [self._new_node("while_body")]
        for stmt in node.body:
            self.visit(stmt)
        # After the loop body, go back to condition:
        for cn in self.current_nodes:
            self.add_edge(cn, loop_cond)
        # False branch (exit):
        loop_exit = self._new_node("while_exit")
        self.add_edge(loop_cond, loop_exit)
        self.current_nodes = [loop_exit]

    def visit_For(self, node: ast.For):
        loop_cond = self._new_node("for (" + ast.unparse(node.target) +
                                   " in " + ast.unparse(node.iter) + ")")
        for cn in self.current_nodes:
            self.add_edge(cn, loop_cond)
        # Loop body:
        self.current_nodes = [self._new_node("for_body")]
        for stmt in node.body:
            self.visit(stmt)
        for cn in self.current_nodes:
            self.add_edge(cn, loop_cond)
        loop_exit = self._new_node("for_exit")
        self.add_edge(loop_cond, loop_exit)
        self.current_nodes = [loop_exit]

def build_cfg_from_source(source: str) -> Dict[str, nx.DiGraph]:
    tree = ast.parse(source)
    cfgs = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            builder = CFGBuilder()
            cfg = builder.build(node)
            cfgs[node.name] = cfg
    return cfgs

# ================================
# Data-Flow Analysis (Reaching Definitions)
# ================================

class DataFlowAnalyzer:
    """
    Performs reaching definitions analysis on a given CFG.
    
    For each CFG node, we extract the set of variable definitions.
    Then, using an iterative algorithm, we compute for each node:
      - in_sets: the definitions reaching the node from its predecessors.
      - out_sets: the definitions after the node.
    
    """
    def __init__(self, cfg: nx.DiGraph):
        self.cfg = cfg
        self.in_sets: Dict[Any, Set[str]] = {}
        self.out_sets: Dict[Any, Set[str]] = {}
        self.node_defs: Dict[Any, Set[str]] = {}  # Definitions made in the node

    def _extract_defs(self, stmts: list) -> Set[str]:
        """Extract variable names from assignment statements in the list of statement strings."""
        defs = set()
        for stmt in stmts:
            try:
                # Parse the dumped statement back to an AST node
                tree = ast.parse(stmt)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                defs.add(target.id)
            except Exception:
                continue
        return defs

    def analyze(self) -> Tuple[Dict[Any, Set[str]], Dict[Any, Set[str]]]:
        # Initialize in and out sets for each node in the CFG
        for node in self.cfg.nodes():
            stmts = self.cfg.nodes[node].get("stmts", [])
            defs = self._extract_defs(stmts)
            self.node_defs[node] = defs
            self.in_sets[node] = set()
            self.out_sets[node] = defs.copy()

        # Iteratively compute reaching definitions until convergence.
        changed = True
        while changed:
            changed = False
            for node in self.cfg.nodes():
                # in[node] is the union of out sets of all predecessor nodes.
                new_in = set()
                for pred in self.cfg.predecessors(node):
                    new_in |= self.out_sets[pred]
                if new_in != self.in_sets[node]:
                    self.in_sets[node] = new_in
                    changed = True
                # For this simple analysis, we assume that any new definition in the node kills
                # any previous definition of the same variable. (Here, kill_set is simply node_defs[node].)
                kill_set = self.node_defs[node]
                new_out = self.node_defs[node] | (new_in - kill_set)
                if new_out != self.out_sets[node]:
                    self.out_sets[node] = new_out
                    changed = True
        return self.in_sets, self.out_sets

def analyze_data_flow(cfg: nx.DiGraph) -> Tuple[Dict[Any, Set[str]], Dict[Any, Set[str]]]:
    """
    Given a CFG (networkx.DiGraph), perform reaching definitions analysis.
    Returns (in_sets, out_sets): mapping each node id to a set of variable names.
    """
    analyzer = DataFlowAnalyzer(cfg)
    return analyzer.analyze()

# ================================
# Helper: Analyze a Python File
# ================================

def analyze_file(file_path: str) -> Dict[str, Any]:
    """
    Given a Python file, parse its source, build CFGs for all function definitions,
    and perform data-flow (reaching definitions) analysis.
    
    Returns a dictionary mapping function names to analysis results:
      {
         "cfg": <networkx.DiGraph>,
         "in_sets": { node_id: set(...), ... },
         "out_sets": { node_id: set(...), ... }
      }
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        source = f.read()
    cfgs = build_cfg_from_source(source)
    results = {}
    for func_name, cfg in cfgs.items():
        in_sets, out_sets = analyze_data_flow(cfg)
        results[func_name] = {
            "cfg": cfg,
            "in_sets": in_sets,
            "out_sets": out_sets,
        }
    return results

# ================================
# Standalone Usage Example
# ================================

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python advanced_analysis.py <python_file>")
        sys.exit(1)
    results = analyze_file(sys.argv[1])
    for func_name, data in results.items():
        print(f"Function: {func_name}")
        print("CFG Nodes:")
        for node, attrs in data["cfg"].nodes(data=True):
            print(f"  Node {node}: {attrs.get('label')}  stmts: {attrs.get('stmts')}")
        print("In Sets:")
        print(data["in_sets"])
        print("Out Sets:")
        print(data["out_sets"])
        print("\n")
