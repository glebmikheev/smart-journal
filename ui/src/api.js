const API_BASE = (import.meta.env.VITE_API_BASE || "").replace(/\/$/, "");

function toApiUrl(path) {
  return `${API_BASE}${path}`;
}

async function request(path, options = {}) {
  const response = await fetch(toApiUrl(path), options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed: ${response.status}`);
  }
  if (response.status === 204) {
    return null;
  }
  return response.json();
}

export function getHealth() {
  return request("/api/health");
}

export function getSelectedProviders() {
  return request("/api/providers/selected");
}

export function listGraphs() {
  return request("/api/graphs");
}

export function createGraph(title) {
  return request("/api/graphs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title })
  });
}

export function listNodes(graphId) {
  return request(`/api/graphs/${encodeURIComponent(graphId)}/nodes`);
}

export function getGraphDetails(graphId) {
  return request(`/api/graphs/${encodeURIComponent(graphId)}/details`);
}

export function createNode(graphId, payload) {
  return request(`/api/graphs/${encodeURIComponent(graphId)}/nodes`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}

export function getNodeDetails(nodeId) {
  return request(`/api/nodes/${encodeURIComponent(nodeId)}/details`);
}

export function createGroup(graphId, name) {
  return request(`/api/graphs/${encodeURIComponent(graphId)}/groups`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name })
  });
}

export function addNodeToGroup(nodeId, groupId) {
  return request(
    `/api/nodes/${encodeURIComponent(nodeId)}/groups/${encodeURIComponent(groupId)}`,
    { method: "POST" }
  );
}

export function removeNodeFromGroup(nodeId, groupId) {
  return request(
    `/api/nodes/${encodeURIComponent(nodeId)}/groups/${encodeURIComponent(groupId)}`,
    { method: "DELETE" }
  );
}

export function createTag(graphId, name) {
  return request(`/api/graphs/${encodeURIComponent(graphId)}/tags`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name })
  });
}

export function addNodeTag(nodeId, tagId) {
  return request(`/api/nodes/${encodeURIComponent(nodeId)}/tags/${encodeURIComponent(tagId)}`, {
    method: "POST"
  });
}

export function removeNodeTag(nodeId, tagId) {
  return request(`/api/nodes/${encodeURIComponent(nodeId)}/tags/${encodeURIComponent(tagId)}`, {
    method: "DELETE"
  });
}

export function searchNodes(query, graphId) {
  const params = new URLSearchParams({ q: query });
  if (graphId) {
    params.set("graph_id", graphId);
  }
  return request(`/api/search?${params.toString()}`);
}

export async function uploadNodeFile(nodeId, file) {
  const form = new FormData();
  form.append("file", file);
  form.append("ingest_now", "true");
  form.append("replay_vector_ops", "true");
  return request(`/api/nodes/${encodeURIComponent(nodeId)}/content-items`, {
    method: "POST",
    body: form
  });
}

export function runVectorQuery(query, topK = 10) {
  return request("/api/vector/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: topK })
  });
}
