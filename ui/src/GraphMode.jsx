import { useEffect, useMemo, useState } from "react";
import GraphCanvas from "./GraphCanvas";

function truncateText(value, limit = 52) {
  const text = String(value || "").trim();
  if (text.length <= limit) {
    return text;
  }
  return `${text.slice(0, Math.max(1, limit - 3)).trimEnd()}...`;
}

function shortId(value) {
  if (typeof value !== "string") {
    return "n/a";
  }
  return value.slice(0, 8);
}

export default function GraphMode({
  selectedGraphId,
  selectedGraph,
  selectedNode,
  selectedNodeId,
  graphGroups,
  graphTags,
  graphTopology,
  ftsQuery,
  setFtsQuery,
  ftsResults,
  semanticQuery,
  setSemanticQuery,
  semanticResults,
  busy,
  handleFulltextSearch,
  handleSemanticSearch,
  focusNode,
  setSelectedVectorResult,
  setViewMode
}) {
  const [selection, setSelection] = useState(null);
  const [visibility, setVisibility] = useState({
    showContainment: true,
    showGroups: true,
    showTags: true,
    showQueries: true
  });

  const graphScene = useMemo(() => {
    if (!selectedGraphId || !graphTopology) {
      return { nodes: [], links: [] };
    }
    const nodeMap = new Map();
    const links = [];
    const linkKeys = new Set();

    const addNode = (item) => {
      if (!item?.id) {
        return;
      }
      if (!nodeMap.has(item.id)) {
        nodeMap.set(item.id, item);
      }
    };

    const addLink = (source, target, kind, extra = {}) => {
      if (!source || !target) {
        return;
      }
      const key = `${kind}:${source}:${target}`;
      if (linkKeys.has(key)) {
        return;
      }
      linkKeys.add(key);
      links.push({ source, target, kind, ...extra });
    };

    const graphKey = `graph:${graphTopology.graph.graph_id}`;
    addNode({
      id: graphKey,
      kind: "graph",
      label: truncateText(graphTopology.graph.title || "Graph", 42),
      subtitle: "catalog",
      color: "#6ea7df"
    });

    for (const node of graphTopology.nodes || []) {
      const nodeKey = `node:${node.node_id}`;
      const groupCount = node.group_ids?.length ?? 0;
      const tagCount = node.tag_ids?.length ?? 0;
      addNode({
        id: nodeKey,
        kind: "node",
        nodeId: node.node_id,
        graphId: node.graph_id,
        label: truncateText(node.title || "(untitled node)", 44),
        subtitle: `${groupCount} groups · ${tagCount} tags`,
        color: "#4f8fce"
      });
      if (visibility.showContainment) {
        addLink(graphKey, nodeKey, "contains");
      }
    }

    if (visibility.showGroups) {
      for (const group of graphTopology.groups || []) {
        addNode({
          id: `group:${group.group_id}`,
          kind: "group",
          groupId: group.group_id,
          label: truncateText(group.name || "group", 36),
          subtitle: `${group.node_count ?? 0} nodes`,
          color: "#42a88b"
        });
      }
      for (const edge of graphTopology.links?.group_membership || []) {
        addLink(`group:${edge.group_id}`, `node:${edge.node_id}`, "group_membership");
      }
    }

    if (visibility.showTags) {
      for (const tag of graphTopology.tags || []) {
        addNode({
          id: `tag:${tag.tag_id}`,
          kind: "tag",
          tagId: tag.tag_id,
          label: truncateText(tag.name || "tag", 36),
          subtitle: `${tag.node_count ?? 0} nodes`,
          color: "#ddb13f"
        });
      }
      for (const edge of graphTopology.links?.tag_membership || []) {
        addLink(`tag:${edge.tag_id}`, `node:${edge.node_id}`, "tag_membership");
      }
    }

    if (visibility.showQueries) {
      const semanticLabel = semanticQuery.trim();
      if (semanticLabel && semanticResults.length) {
        const semanticKey = "query:semantic";
        addNode({
          id: semanticKey,
          kind: "query",
          queryType: "semantic",
          label: "Semantic Query",
          subtitle: truncateText(semanticLabel, 48),
          color: "#e96c63"
        });
        for (const row of semanticResults) {
          const relatedNode = row.node;
          if (!relatedNode?.node_id) {
            continue;
          }
          const targetKey = `node:${relatedNode.node_id}`;
          if (!nodeMap.has(targetKey)) {
            addNode({
              id: targetKey,
              kind: "node",
              nodeId: relatedNode.node_id,
              graphId: relatedNode.graph_id,
              label: truncateText(relatedNode.title || "(external node)", 44),
              subtitle: "from search result",
              color: "#6f9fcb"
            });
          }
          addLink(semanticKey, targetKey, "semantic_query", { score: row.score });
        }
      }

      const ftsLabel = ftsQuery.trim();
      if (ftsLabel && ftsResults.length) {
        const ftsKey = "query:fulltext";
        addNode({
          id: ftsKey,
          kind: "query",
          queryType: "fulltext",
          label: "FTS Query",
          subtitle: truncateText(ftsLabel, 48),
          color: "#8d78f0"
        });
        for (const row of ftsResults) {
          if (!row?.node_id) {
            continue;
          }
          const targetKey = `node:${row.node_id}`;
          if (!nodeMap.has(targetKey)) {
            addNode({
              id: targetKey,
              kind: "node",
              nodeId: row.node_id,
              graphId: row.graph_id,
              label: truncateText(row.title || "(external node)", 44),
              subtitle: "from search result",
              color: "#6f9fcb"
            });
          }
          addLink(ftsKey, targetKey, "fulltext_query");
        }
      }
    }

    return { nodes: [...nodeMap.values()], links };
  }, [selectedGraphId, graphTopology, visibility, semanticQuery, semanticResults, ftsQuery, ftsResults]);

  const selectedNodeGraphKey = selectedNodeId ? `node:${selectedNodeId}` : "";

  useEffect(() => {
    if (!selectedNodeGraphKey) {
      return;
    }
    const node = graphScene.nodes.find((item) => item.id === selectedNodeGraphKey);
    if (node) {
      setSelection(node);
    }
  }, [selectedNodeGraphKey, graphScene.nodes]);

  return (
    <main className="graph-layout">
      <section className="panel panel-wide panel-graph">
        <div className="graph-toolbar">
          <div>
            <h2>2D Knowledge Graph</h2>
            <p className="panel-subtitle">Interactive topology with retrieval overlays for Increment 5+ workflows.</p>
          </div>
          <div className="graph-visibility">
            <button type="button" className={`mode-chip mini ${visibility.showContainment ? "active" : ""}`} onClick={() => setVisibility((prev) => ({ ...prev, showContainment: !prev.showContainment }))}>Containment</button>
            <button type="button" className={`mode-chip mini ${visibility.showGroups ? "active" : ""}`} onClick={() => setVisibility((prev) => ({ ...prev, showGroups: !prev.showGroups }))}>Groups</button>
            <button type="button" className={`mode-chip mini ${visibility.showTags ? "active" : ""}`} onClick={() => setVisibility((prev) => ({ ...prev, showTags: !prev.showTags }))}>Tags</button>
            <button type="button" className={`mode-chip mini ${visibility.showQueries ? "active" : ""}`} onClick={() => setVisibility((prev) => ({ ...prev, showQueries: !prev.showQueries }))}>Query Overlay</button>
          </div>
        </div>
        <div className="graph-meta-strip">
          <div><span className="label">Graph</span><strong>{selectedGraph?.title || "none selected"}</strong></div>
          <div><span className="label">Nodes in canvas</span><strong>{graphScene.nodes.length}</strong></div>
          <div><span className="label">Links in canvas</span><strong>{graphScene.links.length}</strong></div>
          <div><span className="label">Selected node</span><strong>{selectedNode?.title || "none"}</strong></div>
        </div>
        {selectedGraphId ? (
          <GraphCanvas
            graphData={graphScene}
            selectedNodeKey={selectedNodeGraphKey}
            onNodeSelect={(node) => {
              setSelection(node);
              if (node.kind === "node" && node.nodeId) {
                focusNode(node.graphId, node.nodeId);
              }
            }}
          />
        ) : (
          <p className="empty-state">Create or select a graph to render it on the canvas.</p>
        )}
      </section>

      <section className="panel panel-graph-side">
        <h2>Inspector</h2>
        <p className="panel-subtitle">Click any element in the canvas to inspect metadata and jump to the tabular panel.</p>
        <div className="details-grid">
          <div><span className="label">Current Graph ID</span><code>{selectedGraphId || "n/a"}</code></div>
          <div><span className="label">Groups</span><strong>{graphGroups.length}</strong></div>
          <div><span className="label">Tags</span><strong>{graphTags.length}</strong></div>
          <div><span className="label">FTS / Vector</span><strong>{ftsResults.length} / {semanticResults.length}</strong></div>
        </div>
        <div className="inspector-card">
          <h3>Selection</h3>
          {selection ? (
            <>
              <p><strong>{selection.label}</strong></p>
              <p className="panel-subtitle">{selection.subtitle || "No extra context"}</p>
              <code>{selection.id}</code>
              {selection.kind === "node" && selection.nodeId ? (
                <button type="button" onClick={() => { focusNode(selection.graphId, selection.nodeId); setViewMode("control"); }}>
                  Open in control panel
                </button>
              ) : null}
            </>
          ) : (
            <p className="empty-state">No element selected.</p>
          )}
        </div>
        <div className="legend-grid">
          <div className="legend-item"><span className="swatch graph" />Graph</div>
          <div className="legend-item"><span className="swatch node" />Node</div>
          <div className="legend-item"><span className="swatch group" />Group</div>
          <div className="legend-item"><span className="swatch tag" />Tag</div>
          <div className="legend-item"><span className="swatch query" />Query</div>
        </div>
        <form onSubmit={handleFulltextSearch} className="stacked-form compact-form">
          <input value={ftsQuery} onChange={(event) => setFtsQuery(event.target.value)} placeholder="FTS query" />
          <button type="submit" disabled={busy}>Run FTS</button>
        </form>
        <form onSubmit={handleSemanticSearch} className="stacked-form compact-form">
          <input value={semanticQuery} onChange={(event) => setSemanticQuery(event.target.value)} placeholder="Semantic query" />
          <button type="submit" disabled={busy}>Run Vector Query</button>
        </form>
        <div className="results-grid">
          <article>
            <h3>FTS Hits</h3>
            <ul className="compact-list">
              {ftsResults.slice(0, 8).map((row) => (
                <li key={row.node_id}>
                  <button type="button" className="item-row result-row" onClick={() => focusNode(row.graph_id, row.node_id)}>
                    <span>{truncateText(row.title || "Node", 26)}</span>
                    <code>{shortId(row.node_id)}</code>
                  </button>
                </li>
              ))}
              {!ftsResults.length ? <li className="empty-state">No FTS hits.</li> : null}
            </ul>
          </article>
          <article>
            <h3>Vector Hits</h3>
            <ul className="compact-list">
              {semanticResults.slice(0, 8).map((row) => (
                <li key={row.chunk_id || row.external_id}>
                  <button
                    type="button"
                    className="item-row result-row"
                    onClick={() => {
                      setSelectedVectorResult(row);
                      if (row.node?.node_id) {
                        focusNode(row.node.graph_id, row.node.node_id);
                      }
                    }}
                  >
                    <span>{truncateText(row.chunk?.text_preview || row.external_id, 26)}</span>
                    <code>{Number(row.score || 0).toFixed(3)}</code>
                  </button>
                </li>
              ))}
              {!semanticResults.length ? <li className="empty-state">No vector hits.</li> : null}
            </ul>
          </article>
        </div>
      </section>
    </main>
  );
}
