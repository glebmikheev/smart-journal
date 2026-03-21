import { useEffect, useMemo, useState } from "react";
import {
  addNodeTag,
  addNodeToGroup,
  createGraph,
  createGroup,
  createNode,
  createTag,
  getGraphDetails,
  getGraphTopology,
  getHealth,
  getNodeDetails,
  getSelectedProviders,
  listGraphs,
  listNodes,
  removeNodeFromGroup,
  removeNodeTag,
  runVectorQuery,
  searchNodes,
  uploadNodeFile
} from "./api";
import GraphMode from "./GraphMode";

function App() {
  const [viewMode, setViewMode] = useState("graph");
  const [health, setHealth] = useState(null);
  const [providers, setProviders] = useState(null);
  const [graphs, setGraphs] = useState([]);
  const [nodes, setNodes] = useState([]);
  const [graphDetails, setGraphDetails] = useState(null);
  const [graphTopology, setGraphTopology] = useState(null);
  const [nodeDetails, setNodeDetails] = useState(null);
  const [selectedGraphId, setSelectedGraphId] = useState("");
  const [selectedNodeId, setSelectedNodeId] = useState("");
  const [pendingNodeFocusId, setPendingNodeFocusId] = useState("");

  const [graphTitle, setGraphTitle] = useState("");
  const [nodeTitle, setNodeTitle] = useState("");
  const [nodeBody, setNodeBody] = useState("");
  const [groupName, setGroupName] = useState("");
  const [tagName, setTagName] = useState("");
  const [selectedGroupId, setSelectedGroupId] = useState("");
  const [selectedTagId, setSelectedTagId] = useState("");

  const [ftsQuery, setFtsQuery] = useState("");
  const [ftsResults, setFtsResults] = useState([]);
  const [semanticQuery, setSemanticQuery] = useState("");
  const [semanticResults, setSemanticResults] = useState([]);
  const [selectedVectorResult, setSelectedVectorResult] = useState(null);

  const [busy, setBusy] = useState(false);
  const [statusLine, setStatusLine] = useState("Ready");
  const [errorLine, setErrorLine] = useState("");

  const selectedGraph = useMemo(
    () => graphs.find((graph) => graph.graph_id === selectedGraphId) ?? null,
    [graphs, selectedGraphId]
  );

  const selectedNode = useMemo(
    () => nodes.find((node) => node.node_id === selectedNodeId) ?? null,
    [nodes, selectedNodeId]
  );

  const graphGroups = graphDetails?.groups ?? [];
  const graphTags = graphDetails?.tags ?? [];
  const nodeGroups = nodeDetails?.groups ?? [];
  const nodeTags = nodeDetails?.tags ?? [];

  useEffect(() => {
    bootstrap().catch(handleError);
  }, []);

  useEffect(() => {
    if (!selectedGraphId) {
      setNodes([]);
      setNodeDetails(null);
      setGraphDetails(null);
      setGraphTopology(null);
      setSelectedNodeId("");
      return;
    }
    refreshGraphContext(selectedGraphId, pendingNodeFocusId).catch(handleError);
    setPendingNodeFocusId("");
  }, [selectedGraphId]);

  useEffect(() => {
    if (!selectedNodeId) {
      setNodeDetails(null);
      return;
    }
    refreshNodeDetails(selectedNodeId).catch(handleError);
  }, [selectedNodeId]);

  async function bootstrap() {
    setBusy(true);
    setErrorLine("");
    const [healthPayload, providersPayload, graphPayload] = await Promise.all([
      getHealth(),
      getSelectedProviders(),
      listGraphs()
    ]);
    setHealth(healthPayload);
    setProviders(providersPayload);
    setGraphs(graphPayload);
    if (!graphPayload.length) {
      setStatusLine("No graphs yet. Create your first graph.");
      setBusy(false);
      return;
    }
    setSelectedGraphId(graphPayload[0].graph_id);
    setBusy(false);
  }

  async function refreshGraphs(preferredGraphId = "") {
    const payload = await listGraphs();
    setGraphs(payload);
    if (!payload.length) {
      setSelectedGraphId("");
      return;
    }
    const nextGraphId =
      preferredGraphId && payload.some((graph) => graph.graph_id === preferredGraphId)
        ? preferredGraphId
        : payload[0].graph_id;
    setSelectedGraphId(nextGraphId);
  }

  async function refreshGraphContext(graphId, preferredNodeId = "") {
    const [nodesPayload, detailsPayload, topologyPayload] = await Promise.all([
      listNodes(graphId),
      getGraphDetails(graphId),
      getGraphTopology(graphId)
    ]);
    setNodes(nodesPayload);
    setGraphDetails(detailsPayload);
    setGraphTopology(topologyPayload);
    setSelectedGroupId((current) => {
      if (current && detailsPayload.groups.some((group) => group.group_id === current)) {
        return current;
      }
      return detailsPayload.groups[0]?.group_id ?? "";
    });
    setSelectedTagId((current) => {
      if (current && detailsPayload.tags.some((tag) => tag.tag_id === current)) {
        return current;
      }
      return detailsPayload.tags[0]?.tag_id ?? "";
    });
    setSelectedNodeId((current) => {
      if (preferredNodeId && nodesPayload.some((node) => node.node_id === preferredNodeId)) {
        return preferredNodeId;
      }
      if (current && nodesPayload.some((node) => node.node_id === current)) {
        return current;
      }
      return nodesPayload[0]?.node_id ?? "";
    });
  }

  async function refreshNodeDetails(nodeId) {
    const payload = await getNodeDetails(nodeId);
    setNodeDetails(payload);
  }

  async function handleCreateGraph(event) {
    event.preventDefault();
    const title = graphTitle.trim();
    if (!title) {
      return;
    }
    setBusy(true);
    setErrorLine("");
    const graph = await createGraph(title);
    await refreshGraphs(graph.graph_id);
    setGraphTitle("");
    setStatusLine(`Graph "${graph.title}" created.`);
    setBusy(false);
  }

  async function handleCreateNode(event) {
    event.preventDefault();
    if (!selectedGraphId) {
      return;
    }
    const title = nodeTitle.trim();
    if (!title) {
      return;
    }
    setBusy(true);
    setErrorLine("");
    const node = await createNode(selectedGraphId, { title, body: nodeBody });
    setPendingNodeFocusId(node.node_id);
    await refreshGraphContext(selectedGraphId, node.node_id);
    setNodeTitle("");
    setNodeBody("");
    setStatusLine(`Node "${node.title}" created.`);
    setBusy(false);
  }

  async function handleCreateGroup(event) {
    event.preventDefault();
    if (!selectedGraphId) {
      return;
    }
    const name = groupName.trim();
    if (!name) {
      return;
    }
    setBusy(true);
    setErrorLine("");
    await createGroup(selectedGraphId, name);
    await refreshGraphContext(selectedGraphId, selectedNodeId);
    if (selectedNodeId) {
      await refreshNodeDetails(selectedNodeId);
    }
    setGroupName("");
    setStatusLine(`Group "${name}" created.`);
    setBusy(false);
  }

  async function handleCreateTag(event) {
    event.preventDefault();
    if (!selectedGraphId) {
      return;
    }
    const name = tagName.trim();
    if (!name) {
      return;
    }
    setBusy(true);
    setErrorLine("");
    await createTag(selectedGraphId, name);
    await refreshGraphContext(selectedGraphId, selectedNodeId);
    if (selectedNodeId) {
      await refreshNodeDetails(selectedNodeId);
    }
    setTagName("");
    setStatusLine(`Tag "${name}" created.`);
    setBusy(false);
  }

  async function handleAssignGroup() {
    if (!selectedNodeId || !selectedGroupId || !selectedGraphId) {
      return;
    }
    setBusy(true);
    setErrorLine("");
    await addNodeToGroup(selectedNodeId, selectedGroupId);
    await refreshGraphContext(selectedGraphId, selectedNodeId);
    await refreshNodeDetails(selectedNodeId);
    setStatusLine("Group assigned to node.");
    setBusy(false);
  }

  async function handleAssignTag() {
    if (!selectedNodeId || !selectedTagId || !selectedGraphId) {
      return;
    }
    setBusy(true);
    setErrorLine("");
    await addNodeTag(selectedNodeId, selectedTagId);
    await refreshGraphContext(selectedGraphId, selectedNodeId);
    await refreshNodeDetails(selectedNodeId);
    setStatusLine("Tag assigned to node.");
    setBusy(false);
  }

  async function handleRemoveGroup(groupId) {
    if (!selectedNodeId || !selectedGraphId) {
      return;
    }
    setBusy(true);
    setErrorLine("");
    await removeNodeFromGroup(selectedNodeId, groupId);
    await refreshGraphContext(selectedGraphId, selectedNodeId);
    await refreshNodeDetails(selectedNodeId);
    setStatusLine("Group removed from node.");
    setBusy(false);
  }

  async function handleRemoveTag(tagId) {
    if (!selectedNodeId || !selectedGraphId) {
      return;
    }
    setBusy(true);
    setErrorLine("");
    await removeNodeTag(selectedNodeId, tagId);
    await refreshGraphContext(selectedGraphId, selectedNodeId);
    await refreshNodeDetails(selectedNodeId);
    setStatusLine("Tag removed from node.");
    setBusy(false);
  }

  async function handleFulltextSearch(event) {
    event.preventDefault();
    const query = ftsQuery.trim();
    if (!query) {
      return;
    }
    setBusy(true);
    setErrorLine("");
    const results = await searchNodes(query, selectedGraphId || undefined);
    setFtsResults(results);
    setStatusLine(`FTS returned ${results.length} result(s).`);
    setBusy(false);
  }

  async function handleSemanticSearch(event) {
    event.preventDefault();
    const query = semanticQuery.trim();
    if (!query) {
      return;
    }
    setBusy(true);
    setErrorLine("");
    const payload = await runVectorQuery(query, 10);
    setSemanticResults(payload.results || []);
    setSelectedVectorResult(payload.results?.[0] ?? null);
    setStatusLine(`Vector query returned ${payload.results?.length ?? 0} result(s).`);
    setBusy(false);
  }

  async function handleUpload(event) {
    const file = event.target.files?.[0];
    if (!file || !selectedNodeId) {
      return;
    }
    setBusy(true);
    setErrorLine("");
    const payload = await uploadNodeFile(selectedNodeId, file);
    await refreshNodeDetails(selectedNodeId);
    setStatusLine(
      `File "${payload.content_item.filename}" uploaded and ingested (${payload.content_item.extraction_status}).`
    );
    setBusy(false);
    event.target.value = "";
  }

  function focusNode(graphId, nodeId) {
    if (!nodeId) {
      return;
    }
    if (graphId && graphId !== selectedGraphId) {
      setPendingNodeFocusId(nodeId);
      setSelectedGraphId(graphId);
      return;
    }
    setSelectedNodeId(nodeId);
  }

  function handleError(error) {
    setBusy(false);
    setErrorLine(error.message || String(error));
  }

  return (
    <div className="page-shell">
      <header className="hero">
        <p className="hero-kicker">Smart Journal</p>
        <h1>Graph Workspace + Control Panel</h1>
        <p className="hero-subtitle">Local-first knowledge graph orchestration for Increment 5+ with an interactive 2D canvas.</p>
      </header>

      <section className="status-strip">
        <div>
          <span className="label">Status</span>
          <strong>{busy ? "Busy" : "Idle"}</strong>
        </div>
        <div>
          <span className="label">Model</span>
          <strong>{providers?.embedding_provider?.provider_id || "n/a"}</strong>
        </div>
        <div>
          <span className="label">Embedder Warmup</span>
          <strong>{health?.embedding_preload?.ready ? "ready" : "pending/error"}</strong>
        </div>
        <div>
          <span className="label">Bootstrap Replay</span>
          <strong>{health?.bootstrap?.applied_ops ?? 0} ops</strong>
        </div>
        <div className="status-line">{statusLine}</div>
      </section>

      <section className="view-switch">
        <button
          type="button"
          className={`mode-chip ${viewMode === "graph" ? "active" : ""}`}
          onClick={() => setViewMode("graph")}
        >
          Graph View
        </button>
        <button
          type="button"
          className={`mode-chip ${viewMode === "control" ? "active" : ""}`}
          onClick={() => setViewMode("control")}
        >
          Control Panel
        </button>
      </section>

      {errorLine ? <section className="error-banner">{errorLine}</section> : null}

      {viewMode === "graph" ? (
        <GraphMode
          selectedGraphId={selectedGraphId}
          selectedGraph={selectedGraph}
          selectedNode={selectedNode}
          selectedNodeId={selectedNodeId}
          graphGroups={graphGroups}
          graphTags={graphTags}
          graphTopology={graphTopology}
          ftsQuery={ftsQuery}
          setFtsQuery={setFtsQuery}
          ftsResults={ftsResults}
          semanticQuery={semanticQuery}
          setSemanticQuery={setSemanticQuery}
          semanticResults={semanticResults}
          busy={busy}
          handleFulltextSearch={handleFulltextSearch}
          handleSemanticSearch={handleSemanticSearch}
          focusNode={focusNode}
          setSelectedVectorResult={setSelectedVectorResult}
          setViewMode={setViewMode}
        />
      ) : (
      <main className="layout-grid">
        <section className="panel">
          <h2>Graphs</h2>
          <form onSubmit={handleCreateGraph} className="stacked-form">
            <input value={graphTitle} onChange={(event) => setGraphTitle(event.target.value)} placeholder="New graph title" />
            <button type="submit" disabled={busy}>Create graph</button>
          </form>
          <div className="item-list">
            {graphs.map((graph) => (
              <button key={graph.graph_id} className={`item-row ${graph.graph_id === selectedGraphId ? "active" : ""}`} onClick={() => setSelectedGraphId(graph.graph_id)} type="button">
                <span>{graph.title}</span>
                <code>{graph.graph_id.slice(0, 8)}</code>
              </button>
            ))}
            {!graphs.length ? <p className="empty-state">No graphs.</p> : null}
          </div>
        </section>

        <section className="panel">
          <h2>Nodes</h2>
          <p className="panel-subtitle">Graph: <strong>{selectedGraph?.title || "none selected"}</strong></p>
          <form onSubmit={handleCreateNode} className="stacked-form">
            <input value={nodeTitle} onChange={(event) => setNodeTitle(event.target.value)} disabled={!selectedGraphId} placeholder="Node title" />
            <textarea rows={3} value={nodeBody} onChange={(event) => setNodeBody(event.target.value)} disabled={!selectedGraphId} placeholder="Node body" />
            <button type="submit" disabled={!selectedGraphId || busy}>Create node</button>
          </form>
          <div className="item-list">
            {nodes.map((node) => (
              <button key={node.node_id} className={`item-row ${node.node_id === selectedNodeId ? "active" : ""}`} onClick={() => setSelectedNodeId(node.node_id)} type="button">
                <span>{node.title}</span>
                <code>{node.node_id.slice(0, 8)}</code>
              </button>
            ))}
            {!nodes.length ? <p className="empty-state">No nodes.</p> : null}
          </div>
        </section>

        <section className="panel">
          <h2>Retrieval</h2>
          <form onSubmit={handleFulltextSearch} className="stacked-form">
            <input value={ftsQuery} onChange={(event) => setFtsQuery(event.target.value)} placeholder="FTS query (prefix supported)" />
            <button type="submit" disabled={busy}>Run full-text search</button>
          </form>
          <form onSubmit={handleSemanticSearch} className="stacked-form">
            <input value={semanticQuery} onChange={(event) => setSemanticQuery(event.target.value)} placeholder="Semantic query" />
            <button type="submit" disabled={busy}>Run vector query</button>
          </form>
          <div className="results-grid">
            <article>
              <h3>FTS Results</h3>
              <ul>
                {ftsResults.map((row) => (
                  <li key={row.node_id}>
                    <button type="button" className="item-row result-row" onClick={() => focusNode(row.graph_id, row.node_id)}>
                      <span>{row.title}</span>
                      <code>{row.node_id.slice(0, 8)}</code>
                    </button>
                  </li>
                ))}
                {!ftsResults.length ? <li className="empty-state">No FTS results.</li> : null}
              </ul>
            </article>
            <article>
              <h3>Vector Results</h3>
              <ul>
                {semanticResults.map((row) => (
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
                      <span>{row.chunk?.text_preview || row.external_id.slice(0, 8)}</span>
                      <code>{row.score.toFixed(4)}</code>
                    </button>
                  </li>
                ))}
                {!semanticResults.length ? <li className="empty-state">No vector results.</li> : null}
              </ul>
            </article>
          </div>
        </section>

        <section className="panel panel-wide">
          <h2>Graph Details</h2>
          <div className="details-grid">
            <div><span className="label">Graph ID</span><code>{graphDetails?.graph?.graph_id || "n/a"}</code></div>
            <div><span className="label">Created</span><strong>{graphDetails?.graph?.created_at || "n/a"}</strong></div>
            <div><span className="label">Nodes</span><strong>{graphDetails?.nodes?.length ?? 0}</strong></div>
            <div><span className="label">Edges</span><strong>{graphDetails?.edges?.supported ? "enabled" : "planned"}</strong></div>
          </div>
          <p className="panel-subtitle">{graphDetails?.edges?.note}</p>
        </section>

        <section className="panel panel-wide">
          <h2>Node Details</h2>
          <p className="panel-subtitle">Selected node: <strong>{selectedNode?.title || "none selected"}</strong></p>
          <div className="details-grid">
            <div><span className="label">Node ID</span><code>{nodeDetails?.node?.node_id || "n/a"}</code></div>
            <div><span className="label">Created</span><strong>{nodeDetails?.node?.created_at || "n/a"}</strong></div>
            <div><span className="label">Updated</span><strong>{nodeDetails?.node?.updated_at || "n/a"}</strong></div>
            <div><span className="label">Files</span><strong>{nodeDetails?.content_items?.length ?? 0}</strong></div>
          </div>
          <div className="results-grid">
            <article>
              <h3>Attachments</h3>
              <ul>
                {(nodeDetails?.content_items || []).map((item) => (
                  <li key={item.content_item_id}>
                    <strong>{item.filename || "unnamed"}</strong>
                    <code>{item.extraction_status}</code>
                  </li>
                ))}
                {!nodeDetails?.content_items?.length ? <li className="empty-state">No files.</li> : null}
              </ul>
            </article>
            <article>
              <h3>Vector Chunk Preview</h3>
              {selectedVectorResult?.chunk ? (
                <div className="chunk-preview">
                  <p>{selectedVectorResult.chunk.text_preview}</p>
                  <code>{selectedVectorResult.chunk.chunk_id.slice(0, 8)}</code>
                </div>
              ) : (
                <p className="empty-state">Select a vector result to inspect chunk context.</p>
              )}
            </article>
          </div>
        </section>

        <section className="panel panel-wide">
          <h2>Groups & Tags</h2>
          <div className="results-grid">
            <article>
              <h3>Groups</h3>
              <form className="inline-form" onSubmit={handleCreateGroup}>
                <input value={groupName} onChange={(event) => setGroupName(event.target.value)} placeholder="New group name" />
                <button type="submit" disabled={!selectedGraphId || busy}>Add group</button>
              </form>
              <div className="inline-form">
                <select value={selectedGroupId} onChange={(event) => setSelectedGroupId(event.target.value)}>
                  <option value="">Select group</option>
                  {graphGroups.map((group) => (
                    <option key={group.group_id} value={group.group_id}>{group.name}</option>
                  ))}
                </select>
                <button type="button" onClick={() => handleAssignGroup().catch(handleError)} disabled={!selectedNodeId || !selectedGroupId || busy}>Assign to node</button>
              </div>
              <ul>
                {nodeGroups.map((group) => (
                  <li key={group.group_id}>
                    <strong>{group.name}</strong>
                    <button type="button" onClick={() => handleRemoveGroup(group.group_id).catch(handleError)}>remove</button>
                  </li>
                ))}
                {!nodeGroups.length ? <li className="empty-state">Node has no groups.</li> : null}
              </ul>
            </article>
            <article>
              <h3>Tags</h3>
              <form className="inline-form" onSubmit={handleCreateTag}>
                <input value={tagName} onChange={(event) => setTagName(event.target.value)} placeholder="New tag name" />
                <button type="submit" disabled={!selectedGraphId || busy}>Add tag</button>
              </form>
              <div className="inline-form">
                <select value={selectedTagId} onChange={(event) => setSelectedTagId(event.target.value)}>
                  <option value="">Select tag</option>
                  {graphTags.map((tag) => (
                    <option key={tag.tag_id} value={tag.tag_id}>{tag.name}</option>
                  ))}
                </select>
                <button type="button" onClick={() => handleAssignTag().catch(handleError)} disabled={!selectedNodeId || !selectedTagId || busy}>Assign to node</button>
              </div>
              <ul>
                {nodeTags.map((tag) => (
                  <li key={tag.tag_id}>
                    <strong>{tag.name}</strong>
                    <button type="button" onClick={() => handleRemoveTag(tag.tag_id).catch(handleError)}>remove</button>
                  </li>
                ))}
                {!nodeTags.length ? <li className="empty-state">Node has no tags.</li> : null}
              </ul>
            </article>
          </div>
        </section>

        <section className="panel panel-wide">
          <h2>Ingestion</h2>
          <p className="panel-subtitle">Selected node: <strong>{selectedNode?.title || "none selected"}</strong></p>
          <label className="file-drop">
            <input type="file" disabled={!selectedNodeId || busy} onChange={(event) => { handleUpload(event).catch(handleError); }} />
            <span>Drop file or click to upload and ingest immediately</span>
          </label>
        </section>
      </main>
      )}
    </div>
  );
}

export default App;
