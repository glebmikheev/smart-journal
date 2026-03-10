import { useEffect, useMemo, useState } from "react";
import {
  createGraph,
  createNode,
  getHealth,
  getSelectedProviders,
  listGraphs,
  listNodes,
  runVectorQuery,
  searchNodes,
  uploadNodeFile
} from "./api";

function App() {
  const [health, setHealth] = useState(null);
  const [providers, setProviders] = useState(null);
  const [graphs, setGraphs] = useState([]);
  const [nodes, setNodes] = useState([]);
  const [selectedGraphId, setSelectedGraphId] = useState("");
  const [selectedNodeId, setSelectedNodeId] = useState("");
  const [graphTitle, setGraphTitle] = useState("");
  const [nodeTitle, setNodeTitle] = useState("");
  const [nodeBody, setNodeBody] = useState("");
  const [ftsQuery, setFtsQuery] = useState("");
  const [ftsResults, setFtsResults] = useState([]);
  const [semanticQuery, setSemanticQuery] = useState("");
  const [semanticResults, setSemanticResults] = useState([]);
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

  useEffect(() => {
    bootstrap().catch(handleError);
  }, []);

  useEffect(() => {
    if (!selectedGraphId) {
      setNodes([]);
      setSelectedNodeId("");
      return;
    }
    refreshNodes(selectedGraphId).catch(handleError);
  }, [selectedGraphId]);

  async function bootstrap() {
    setBusy(true);
    setErrorLine("");
    const [healthPayload, providersPayload] = await Promise.all([
      getHealth(),
      getSelectedProviders()
    ]);
    setHealth(healthPayload);
    setProviders(providersPayload);
    await refreshGraphs();
    setBusy(false);
  }

  async function refreshGraphs() {
    const payload = await listGraphs();
    setGraphs(payload);
    if (!payload.length) {
      setSelectedGraphId("");
      setStatusLine("No graphs yet. Create your first graph.");
      return;
    }
    setSelectedGraphId((current) => {
      if (current && payload.some((graph) => graph.graph_id === current)) {
        return current;
      }
      return payload[0].graph_id;
    });
  }

  async function refreshNodes(graphId) {
    const payload = await listNodes(graphId);
    setNodes(payload);
    setSelectedNodeId((current) => {
      if (current && payload.some((node) => node.node_id === current)) {
        return current;
      }
      return payload[0]?.node_id ?? "";
    });
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
    await refreshGraphs();
    setSelectedGraphId(graph.graph_id);
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
    await refreshNodes(selectedGraphId);
    setSelectedNodeId(node.node_id);
    setNodeTitle("");
    setNodeBody("");
    setStatusLine(`Node "${node.title}" created.`);
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
    setStatusLine(
      `File "${payload.content_item.filename}" uploaded and ingested (${payload.content_item.extraction_status}).`
    );
    await refreshNodes(selectedGraphId);
    setBusy(false);
    event.target.value = "";
  }

  function handleError(error) {
    setBusy(false);
    setErrorLine(error.message || String(error));
  }

  return (
    <div className="page-shell">
      <header className="hero">
        <p className="hero-kicker">Smart Journal</p>
        <h1>FastAPI + React control panel</h1>
        <p className="hero-subtitle">
          Local-first knowledge graph orchestration for Increment 5+.
        </p>
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
          <span className="label">Bootstrap Replay</span>
          <strong>{health?.bootstrap?.applied_ops ?? 0} ops</strong>
        </div>
        <div className="status-line">{statusLine}</div>
      </section>

      {errorLine ? <section className="error-banner">{errorLine}</section> : null}

      <main className="layout-grid">
        <section className="panel">
          <h2>Graphs</h2>
          <form onSubmit={handleCreateGraph} className="stacked-form">
            <input
              type="text"
              placeholder="New graph title"
              value={graphTitle}
              onChange={(event) => setGraphTitle(event.target.value)}
            />
            <button type="submit" disabled={busy}>
              Create graph
            </button>
          </form>
          <div className="item-list">
            {graphs.map((graph) => (
              <button
                key={graph.graph_id}
                className={`item-row ${graph.graph_id === selectedGraphId ? "active" : ""}`}
                onClick={() => setSelectedGraphId(graph.graph_id)}
                type="button"
              >
                <span>{graph.title}</span>
                <code>{graph.graph_id.slice(0, 8)}</code>
              </button>
            ))}
            {!graphs.length ? <p className="empty-state">No graphs.</p> : null}
          </div>
        </section>

        <section className="panel">
          <h2>Nodes</h2>
          <p className="panel-subtitle">
            Graph: <strong>{selectedGraph?.title || "none selected"}</strong>
          </p>
          <form onSubmit={handleCreateNode} className="stacked-form">
            <input
              type="text"
              placeholder="Node title"
              value={nodeTitle}
              onChange={(event) => setNodeTitle(event.target.value)}
              disabled={!selectedGraphId}
            />
            <textarea
              placeholder="Node body"
              value={nodeBody}
              onChange={(event) => setNodeBody(event.target.value)}
              disabled={!selectedGraphId}
              rows={4}
            />
            <button type="submit" disabled={!selectedGraphId || busy}>
              Create node
            </button>
          </form>
          <div className="item-list">
            {nodes.map((node) => (
              <button
                key={node.node_id}
                className={`item-row ${node.node_id === selectedNodeId ? "active" : ""}`}
                onClick={() => setSelectedNodeId(node.node_id)}
                type="button"
              >
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
            <input
              type="text"
              placeholder="FTS query"
              value={ftsQuery}
              onChange={(event) => setFtsQuery(event.target.value)}
            />
            <button type="submit" disabled={busy}>
              Run full-text search
            </button>
          </form>
          <form onSubmit={handleSemanticSearch} className="stacked-form">
            <input
              type="text"
              placeholder="Semantic query"
              value={semanticQuery}
              onChange={(event) => setSemanticQuery(event.target.value)}
            />
            <button type="submit" disabled={busy}>
              Run vector query
            </button>
          </form>
          <div className="results-grid">
            <article>
              <h3>FTS Results</h3>
              <ul>
                {ftsResults.map((row) => (
                  <li key={row.node_id}>
                    <strong>{row.title}</strong>
                    <code>{row.node_id.slice(0, 8)}</code>
                  </li>
                ))}
                {!ftsResults.length ? <li className="empty-state">No FTS results.</li> : null}
              </ul>
            </article>
            <article>
              <h3>Vector Results</h3>
              <ul>
                {semanticResults.map((row) => (
                  <li key={row.external_id}>
                    <strong>{row.external_id.slice(0, 8)}</strong>
                    <code>{row.score.toFixed(4)}</code>
                  </li>
                ))}
                {!semanticResults.length ? (
                  <li className="empty-state">No vector results.</li>
                ) : null}
              </ul>
            </article>
          </div>
        </section>

        <section className="panel panel-wide">
          <h2>Ingestion</h2>
          <p className="panel-subtitle">
            Selected node: <strong>{selectedNode?.title || "none selected"}</strong>
          </p>
          <label className="file-drop">
            <input
              type="file"
              disabled={!selectedNodeId || busy}
              onChange={(event) => {
                handleUpload(event).catch(handleError);
              }}
            />
            <span>Drop file or click to upload and ingest immediately</span>
          </label>
        </section>
      </main>
    </div>
  );
}

export default App;
