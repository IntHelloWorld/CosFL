import re
from dataclasses import dataclass
from typing import Optional

import igraph as ig
import leidenalg as la
import networkx as nx


@dataclass
class CGMethodNode:
    package: str
    class_name: str
    method_name: str
    start_line: int
    end_line: int
    comment: Optional[str] = None
    source: Optional[str] = None

    def __post_init__(self):
        self.signature = f"{self.package}@{self.class_name}:{self.method_name}({self.start_line}-{self.end_line})"

    def __hash__(self):
        return hash(self.signature)

    def __eq__(self, other):
        return self.signature == other.signature


def callstack_to_graph(callstack_files):
    """
    convert callstack file to networkx graph
    """
    pattern = r"\[(\d+)\]#(.*?)#(.*?)@(.*?):(.*?)\((\d+)-(\d+)\)"
    call_graph = nx.DiGraph()
    method_stack = []

    for callstack_file in callstack_files:
        with open(callstack_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                match = re.match(pattern, line)
                if not match:
                    raise ValueError(
                        f"Error: Invalid line in callstack file: {line}"
                    )
                (
                    level,
                    callsite,
                    package,
                    class_name,
                    method_name,
                    start_line,
                    end_line,
                ) = match.groups()
                level = int(level)
                if "$" in method_name:  # this is for some inner constructors
                    method_name = method_name.split("$")[-1]
                node = CGMethodNode(
                    package,
                    class_name,
                    method_name,
                    int(start_line),
                    int(end_line),
                )
                if len(method_stack) > level:
                    method_stack = method_stack[:level]
                method_stack.append(node)
                call_graph.add_node(node)
                if level != 0:
                    prev_node = method_stack[level - 1]
                    if call_graph.has_edge(prev_node, node):
                        call_graph[prev_node][node]["weight"] += 1
                    else:
                        call_graph.add_edge(prev_node, node, weight=1)
    return call_graph


def load_callgraph(graphml_files):
    """
    load the callstack from graphml file
    """

    def get_node_from_str(node_str):
        pattern = r"(.*?)@(.*?):(.*?)\((\d+)-(\d+)\)"
        match = re.match(pattern, node_str)
        if not match:
            raise ValueError(f"Error: Invalid node string: {node_str}")
        package, class_name, method_name, start_line, end_line = match.groups()
        return CGMethodNode(
            package, class_name, method_name, int(start_line), int(end_line)
        )

    new_G = nx.DiGraph()
    for graphml_file in graphml_files:
        raw_G = nx.read_gml(graphml_file)
        for u, v, data in raw_G.edges(data=True):
            new_G.add_edge(
                get_node_from_str(u),
                get_node_from_str(v),
                weight=int(data["weight"]),
            )
    return new_G


def cluster_graph(G: nx.DiGraph, min_size: int = 5, max_size: int = 15):
    """
    Cluster the graph using the Leiden algorithm and extract all sub-graphs in networkx format,
    constraining community sizes between min_size and max_size.
    """
    # Convert NetworkX graph to igraph
    ig_graph = ig.Graph.from_networkx(G)

    # Create a mapping of igraph vertex indices to original NetworkX nodes
    ig_to_nx_mapping = {i: node for i, node in enumerate(G.nodes())}

    # Get edge weights from the original graph
    weights = [G[edge[0]][edge[1]]["weight"] for edge in G.edges()]

    # Apply Leiden algorithm
    partition = la.find_partition(
        ig_graph,
        la.ModularityVertexPartition,
        weights=weights,
        max_comm_size=max_size,
        seed=123,
    )

    # Get cluster assignments
    clusters = partition.membership

    # Calculate the sizes of each community
    community_sizes = {
        cluster_id: clusters.count(cluster_id) for cluster_id in set(clusters)
    }

    # Identify small communities
    small_communities = [
        c for c, size in community_sizes.items() if size < min_size
    ]

    for small_comm in small_communities:
        # Get nodes in the small community
        small_comm_nodes = [
            i for i, cluster in enumerate(clusters) if cluster == small_comm
        ]

        # Find the nearest larger community
        nearest_comm = None
        max_connections = 0
        for target_comm in set(clusters):
            if (
                target_comm != small_comm
                and community_sizes[target_comm] >= min_size
            ):
                weights = []
                for node in small_comm_nodes:
                    for neighbor in ig_graph.neighbors(node):
                        if clusters[neighbor] == target_comm:
                            try:
                                weights.append(
                                    ig_graph.es[
                                        ig_graph.get_eid(node, neighbor)
                                    ]["weight"]
                                )
                            except:
                                weights.append(
                                    ig_graph.es[
                                        ig_graph.get_eid(neighbor, node)
                                    ]["weight"]
                                )
                connections = sum(weights)
                if connections > max_connections:
                    max_connections = connections
                    nearest_comm = target_comm

        # Merge the small community into the nearest larger community
        if nearest_comm is not None:
            for node in small_comm_nodes:
                clusters[node] = nearest_comm

            # Update community sizes
            community_sizes[nearest_comm] += community_sizes[small_comm]
            del community_sizes[small_comm]

    # Create subgraphs based on clusters
    subgraphs = []
    for cluster_id in set(clusters):
        nodes = [
            ig_to_nx_mapping[i]
            for i, cluster in enumerate(clusters)
            if cluster == cluster_id
        ]
        if len(nodes) >= min_size:
            subgraph = G.subgraph(nodes)
            subgraphs.append(subgraph)

    # If no clusters were found, return the original graph
    if len(subgraphs) == 0:
        subgraphs = [G]

    return subgraphs


def process_comment(comment: str):
    """
    simplify the comment, e.g.:
    """
    new_lines = []
    lines = comment.split("\n")
    for line in lines:
        cleaned_line = line.lstrip("/* ").strip()
        if cleaned_line and not cleaned_line.startswith("@"):
            new_lines.append(cleaned_line)
    return " ".join(new_lines)


def subgraph_to_text(subgraph: nx.DiGraph):
    """
    convert the subgraph to input text for the LLM
    """
    input_text = (
        "\nMethods\n\nid|className:methodName(startLine-endLine)|comment\n"
    )
    for i, node in enumerate(subgraph.nodes):
        comment = process_comment(node.comment) if node.comment else ""
        input_text += f"{i}|{node.class_name}:{node.method_name}({node.start_line}-{node.end_line})|{comment}\n"
    input_text += "\nCalls\n\nid|source|target\n"
    for i, (u, v) in enumerate(subgraph.edges):
        input_text += f"{i}|{u.class_name}:{u.method_name}({u.start_line}-{u.end_line})|{v.class_name}:{v.method_name}({v.start_line}-{v.end_line})\n"
    return input_text


def cg_summary_to_text(json_data):
    """
    convert the cg summary to text
    """
    # Extract main components
    title = json_data["title"]
    summary = json_data["summary"]
    findings = json_data["findings"]

    # Construct the text output
    text_output = f"- Title {title}\n\n"
    text_output += f"- Summary {summary}\n\n"
    text_output += "- Findings:\n"

    # Add each finding
    for i, finding in enumerate(findings, 1):
        text_output += f"    {i}. {finding['summary']}\n"
        text_output += f"        {finding['explanation']}\n\n"

    return text_output.strip()


def prepare_method_summarization_input(method_node, method_context_node):
    """
    prepare the input for the method summarization
    """
    context_text = (
        "Unavailable"
        if method_context_node is None
        else method_context_node.text
    )
    input_text = f"\nMethod Code:\n\n{method_node.text}\n"
    input_text += f"\nDeveloper Comment:\n\n{process_comment(method_node.metadata['comment'])}\n"
    input_text += f"\nModule Context:\n\n{context_text}\n"
    return input_text


if __name__ == "__main__":
    G = load_callgraph("CallGraph/resource/callgraph.graphml")
    print("over")
