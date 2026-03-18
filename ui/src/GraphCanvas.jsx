import { useEffect, useMemo, useRef } from "react";
import ForceGraph2D from "react-force-graph-2d";

function nodeRadius(node) {
  const base = {
    graph: 11,
    node: 7,
    group: 9,
    tag: 8,
    query: 10
  };
  return base[node.kind] ?? 7;
}

function linkColor(link) {
  const colors = {
    contains: "rgba(122, 153, 193, 0.26)",
    group_membership: "rgba(66, 168, 139, 0.55)",
    tag_membership: "rgba(223, 179, 63, 0.55)",
    semantic_query: "rgba(233, 108, 99, 0.62)",
    fulltext_query: "rgba(138, 121, 243, 0.62)"
  };
  return colors[link.kind] ?? "rgba(142, 163, 192, 0.4)";
}

export default function GraphCanvas({
  graphData,
  selectedNodeKey,
  onNodeSelect,
  className = ""
}) {
  const graphRef = useRef(null);

  const data = useMemo(
    () => ({
      nodes: graphData?.nodes ?? [],
      links: graphData?.links ?? []
    }),
    [graphData]
  );

  useEffect(() => {
    if (!selectedNodeKey || !graphRef.current) {
      return;
    }
    const target = data.nodes.find((node) => node.id === selectedNodeKey);
    if (!target || typeof target.x !== "number" || typeof target.y !== "number") {
      return;
    }
    graphRef.current.centerAt(target.x, target.y, 600);
    graphRef.current.zoom(2.3, 600);
  }, [selectedNodeKey, data.nodes]);

  return (
    <div className={`graph-canvas ${className}`.trim()}>
      <ForceGraph2D
        ref={graphRef}
        graphData={data}
        cooldownTicks={120}
        d3AlphaDecay={0.028}
        d3VelocityDecay={0.22}
        linkColor={linkColor}
        linkWidth={(link) => (link.kind?.includes("query") ? 1.7 : 1.1)}
        linkDirectionalParticles={(link) => (link.kind?.includes("query") ? 2 : 0)}
        linkDirectionalParticleWidth={1.6}
        linkDirectionalParticleSpeed={(link) => (link.kind?.includes("query") ? 0.0035 : 0)}
        onNodeClick={(node) => onNodeSelect?.(node)}
        nodeLabel={(node) => `${node.label}\n${node.subtitle || ""}`.trim()}
        nodeCanvasObject={(node, ctx, globalScale) => {
          const radius = nodeRadius(node);
          const selected = node.id === selectedNodeKey;
          const ring = selected ? 3 : 0;
          const fillColor = node.color || "#79a8df";

          ctx.beginPath();
          ctx.arc(node.x, node.y, radius + ring, 0, 2 * Math.PI, false);
          ctx.fillStyle = selected ? "rgba(255, 246, 209, 0.95)" : fillColor;
          ctx.fill();

          ctx.beginPath();
          ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI, false);
          ctx.fillStyle = fillColor;
          ctx.fill();
          ctx.strokeStyle = selected ? "rgba(12, 20, 32, 0.95)" : "rgba(12, 20, 32, 0.45)";
          ctx.lineWidth = selected ? 2 : 1;
          ctx.stroke();

          if (globalScale < 1.25 && !selected) {
            return;
          }
          const fontSize = selected ? 13 : 11;
          ctx.font = `${fontSize / globalScale}px "Space Grotesk", sans-serif`;
          ctx.fillStyle = "rgba(242, 247, 255, 0.95)";
          ctx.textAlign = "center";
          ctx.textBaseline = "top";
          ctx.fillText(node.label, node.x, node.y + radius + 2.8);
        }}
      />
    </div>
  );
}
