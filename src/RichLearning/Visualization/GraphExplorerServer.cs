// ═══════════════════════════════════════════════════════════════════════════
//  Rich Learning Base — Graph Explorer (Multi-Level)
//
//  Embedded HTTP server + Cytoscape.js visualisation with support for
//  hierarchical meta-levels:
//
//    Level 0  — Base landmarks (individual states)
//    Meta-1   — Scout / cluster-level abstractions
//    Meta-2   — Region-level abstractions
//    Meta-3   — Domain-level abstractions
//
//  Reads through IGraphMemory so it works identically with any backend:
//    - InMemoryGraphMemory (default for PoCs)
//    - LiteDB (embedded, zero-setup)
//    - Neo4j (server-mode, production)
//    - Any custom IGraphMemory implementation
//
//  Zero external dependencies — uses only System.Net.HttpListener.
//
//  Endpoints:
//    GET /                   → HTML SPA
//    GET /api/graph?level=N  → nodes + edges at hierarchy level N (default 0)
//    GET /api/levels         → available levels with counts
// ═══════════════════════════════════════════════════════════════════════════

using System.Net;
using System.Diagnostics;
using System.Text;
using System.Text.Json;
using RichLearning.Abstractions;

namespace RichLearning.Visualization;

/// <summary>
/// Embedded Graph Explorer server for visualising topological graph memory.
///
/// Provides a browser-based interactive visualisation of the Rich Learning
/// topological graph, supporting hierarchical meta-levels.
///
/// Usage:
///   var memory = new InMemoryGraphMemory();
///   // ... populate graph ...
///   await GraphExplorerServer.RunAsync(memory);
///
/// This is domain-agnostic: works with any IGraphMemory implementation.
/// </summary>
public static class GraphExplorerServer
{
    private static readonly JsonSerializerOptions s_json = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase
    };

    /// <summary>
    /// Starts the Graph Explorer HTTP server, opens the default browser,
    /// and blocks until the user presses Ctrl+C.
    /// </summary>
    /// <param name="memory">The graph memory to visualise.</param>
    /// <param name="port">HTTP port to bind (default: 5080).</param>
    public static async Task RunAsync(IGraphMemory memory, int port = 5080)
    {
        var listener = new HttpListener();
        listener.Prefixes.Add($"http://localhost:{port}/");

        try { listener.Start(); }
        catch (Exception ex)
        {
            Console.WriteLine($"  Cannot bind port {port}: {ex.Message}");
            return;
        }

        Console.WriteLine($"  Graph Explorer → http://localhost:{port}");
        Console.WriteLine("  Press Ctrl+C to stop.\n");

        try
        {
            Process.Start(new ProcessStartInfo($"http://localhost:{port}")
                { UseShellExecute = true });
        }
        catch { /* browser open is non-critical */ }

        using var cts = new CancellationTokenSource();
        Console.CancelKeyPress += (_, e) => { e.Cancel = true; cts.Cancel(); };

        try
        {
            while (!cts.IsCancellationRequested)
            {
                var ctx = await listener.GetContextAsync().WaitAsync(cts.Token);
                _ = HandleAsync(ctx, memory);
            }
        }
        catch (OperationCanceledException) { }
        finally
        {
            listener.Stop();
            listener.Close();
        }

        Console.WriteLine("  Graph Explorer stopped.");
    }

    // ── Request Router ────────────────────────────────────────────────

    private static async Task HandleAsync(
        HttpListenerContext ctx, IGraphMemory mem)
    {
        try
        {
            switch (ctx.Request.Url?.AbsolutePath)
            {
                case "/" or null:
                    Reply(ctx, "text/html", Html);
                    break;
                case "/api/graph":
                    await ServeGraph(ctx, mem);
                    break;
                case "/api/levels":
                    await ServeLevels(ctx, mem);
                    break;
                default:
                    ctx.Response.StatusCode = 404;
                    break;
            }
        }
        catch (Exception ex)
        {
            ctx.Response.StatusCode = 500;
            Reply(ctx, "application/json",
                JsonSerializer.Serialize(new { error = ex.Message }));
        }
        finally { ctx.Response.Close(); }
    }

    // ── API: GET /api/levels ──────────────────────────────────────────

    private static async Task ServeLevels(
        HttpListenerContext ctx, IGraphMemory mem)
    {
        var all = await mem.GetAllLandmarksAsync();
        var levelCounts = all
            .GroupBy(l => l.HierarchyLevel)
            .OrderBy(g => g.Key)
            .Select(g => new { level = g.Key, count = g.Count(), name = LevelName(g.Key) })
            .ToList();

        Reply(ctx, "application/json",
            JsonSerializer.Serialize(new { levels = levelCounts }, s_json));
    }

    // ── API: GET /api/graph?level=N ───────────────────────────────────

    private static async Task ServeGraph(
        HttpListenerContext ctx, IGraphMemory mem)
    {
        var query = ctx.Request.Url?.Query ?? "";
        int level = 0;
        if (!string.IsNullOrEmpty(query))
        {
            var match = System.Text.RegularExpressions.Regex.Match(query, @"level=(\d+)");
            if (match.Success) int.TryParse(match.Groups[1].Value, out level);
        }

        var allLandmarks = await mem.GetAllLandmarksAsync();
        var landmarks = allLandmarks.Where(l => l.HierarchyLevel == level).ToList();

        if (landmarks.Count == 0 && level == 0)
            landmarks = allLandmarks.ToList(); // fallback: show everything

        var landmarkIds = landmarks.Select(l => l.Id).ToHashSet();
        var topK = Math.Max(1, landmarks.Count / 5);
        var frontierIds = landmarks.Count > 0
            ? (await mem.GetFrontierLandmarksAsync(topK))
                .Where(f => landmarkIds.Contains(f.Id))
                .Select(f => f.Id).ToHashSet()
            : new HashSet<string>();

        var nodes = new List<object>(landmarks.Count);
        var edges = new List<object>();

        foreach (var lm in landmarks)
        {
            nodes.Add(new
            {
                lm.Id,
                lm.VisitCount,
                lm.ValueEstimate,
                lm.NoveltyScore,
                lm.ClusterId,
                lm.HierarchyLevel,
                childCount = lm.ChildNodeIds.Count,
                isFrontier = frontierIds.Contains(lm.Id),
                lm.CreatedTimestep,
                lm.LastVisitedTimestep
            });

            foreach (var t in await mem.GetOutgoingTransitionsAsync(lm.Id))
            {
                if (!landmarkIds.Contains(t.TargetId)) continue;
                edges.Add(new
                {
                    source = t.SourceId,
                    target = t.TargetId,
                    t.Action,
                    t.Reward,
                    t.TransitionCount,
                    t.SuccessRate
                });
            }
        }

        var (lc, tc) = await mem.GetGraphStatsAsync();

        // Compute per-level stats
        var levelCounts = allLandmarks
            .GroupBy(l => l.HierarchyLevel)
            .ToDictionary(g => g.Key, g => g.Count());

        // Aggregate stats for the current level
        var visits = landmarks.Select(l => (double)l.VisitCount).ToList();
        var values = landmarks.Select(l => l.ValueEstimate).ToList();
        var novelties = landmarks.Select(l => l.NoveltyScore).ToList();
        var clusterIds = landmarks.Select(l => l.ClusterId).Distinct().ToList();
        var frontierCount = frontierIds.Count;
        var totalChildren = landmarks.Sum(l => l.ChildNodeIds.Count);

        // Edge aggregates
        var allEdges = new List<(int action, double reward, int count, double sr)>();
        foreach (var lm in landmarks)
            foreach (var t in await mem.GetOutgoingTransitionsAsync(lm.Id))
                if (landmarkIds.Contains(t.TargetId))
                    allEdges.Add((t.Action, t.Reward, t.TransitionCount, t.SuccessRate));

        var distinctActions = allEdges.Select(e => e.action).Distinct().ToList();

        var aggregate = new
        {
            nodeCount = landmarks.Count,
            edgeCount = allEdges.Count,
            frontierCount,
            clusterCount = clusterIds.Count,
            clusterIds,
            totalChildren,
            visitSum = (long)visits.Sum(),
            visitAvg = visits.Count > 0 ? visits.Average() : 0,
            visitMin = visits.Count > 0 ? (long)visits.Min() : 0,
            visitMax = visits.Count > 0 ? (long)visits.Max() : 0,
            valueAvg = values.Count > 0 ? values.Average() : 0,
            valueMin = values.Count > 0 ? values.Min() : 0,
            valueMax = values.Count > 0 ? values.Max() : 0,
            noveltyAvg = novelties.Count > 0 ? novelties.Average() : 0,
            noveltyMin = novelties.Count > 0 ? novelties.Min() : 0,
            noveltyMax = novelties.Count > 0 ? novelties.Max() : 0,
            timestepMin = landmarks.Count > 0 ? landmarks.Min(l => l.CreatedTimestep) : 0,
            timestepMax = landmarks.Count > 0 ? landmarks.Max(l => l.LastVisitedTimestep) : 0,
            edgeTotalTraversals = allEdges.Sum(e => e.count),
            edgeAvgReward = allEdges.Count > 0 ? allEdges.Average(e => e.reward) : 0,
            edgeAvgSuccessRate = allEdges.Count > 0 ? allEdges.Average(e => e.sr) : 0,
            distinctActionCount = distinctActions.Count,
            distinctActions
        };

        Reply(ctx, "application/json",
            JsonSerializer.Serialize(new
            {
                nodes,
                edges,
                currentLevel = level,
                levelName = LevelName(level),
                stats = new { landmarks = lc, transitions = tc },
                levelCounts,
                aggregate
            }, s_json));
    }

    // ── Helpers ───────────────────────────────────────────────────────

    private static string LevelName(int level) => level switch
    {
        0 => "Base",
        1 => "Meta-1 (Scout)",
        2 => "Meta-2 (Region)",
        3 => "Meta-3 (Domain)",
        _ => $"Level {level}"
    };

    private static void Reply(
        HttpListenerContext ctx, string contentType, string body)
    {
        var buf = Encoding.UTF8.GetBytes(body);
        ctx.Response.ContentType = contentType + "; charset=utf-8";
        ctx.Response.ContentLength64 = buf.Length;
        ctx.Response.OutputStream.Write(buf);
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Embedded single-page visualisation (Cytoscape.js + dark theme)
    //  Supports hierarchical level switching with visual differentiation
    // ═══════════════════════════════════════════════════════════════════

    private const string Html = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Rich Learning — Graph Explorer</title>
<script src="https://cdn.jsdelivr.net/npm/cytoscape@3.30.4/dist/cytoscape.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:system-ui,-apple-system,sans-serif;background:#0d1117;
     color:#e6edf3;height:100vh;display:flex;flex-direction:column}
header{background:#161b22;border-bottom:1px solid #30363d;
       padding:10px 20px;display:flex;align-items:center;gap:14px;flex-wrap:wrap}
header h1{font-size:17px;font-weight:600;
          background:linear-gradient(135deg,#58a6ff,#3fb950);
          -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.stats{display:flex;gap:18px;margin-left:auto;font-size:12px;color:#8b949e}
.stats b{color:#e6edf3}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:4px}
#main{display:flex;flex:1;overflow:hidden;position:relative}
#cy{flex:1}
/* ── Side panels ── */
.side-panel{width:300px;background:#161b22;overflow-y:auto;
            display:none;font-size:13px;flex-shrink:0}
.side-panel.open{display:block}
.side-panel h2{font-size:11px;color:#8b949e;text-transform:uppercase;
               letter-spacing:.5px;margin:14px 0 8px;padding:0 16px}
.side-panel h2:first-child{margin-top:0;padding-top:16px}
#props{border-right:1px solid #30363d}
#panel{border-left:1px solid #30363d;padding:16px}
#panel h2{padding:0;margin-bottom:10px}
.row{display:flex;justify-content:space-between;padding:6px 16px;
     border-bottom:1px solid #21262d}
#panel .row{padding:7px 0}
.row .k{color:#8b949e;font-size:12px}
.row .v{font-weight:500;font-size:12px;text-align:right;max-width:160px;
        overflow:hidden;text-overflow:ellipsis}
.section-sep{border-top:1px solid #30363d;margin:6px 0}
.prop-toggle{background:#21262d;color:#c9d1d9;border:1px solid #30363d;
             padding:5px 12px;border-radius:5px;cursor:pointer;font-size:11px}
.prop-toggle:hover{background:#30363d}
.prop-toggle.active{background:#1f6feb;color:#fff;border-color:#388bfd}

/* ── Level selector bar ── */
.level-bar{display:flex;gap:6px;align-items:center}
.level-bar label{font-size:11px;color:#8b949e;margin-right:4px}
.level-btn{background:#21262d;color:#8b949e;border:1px solid #30363d;
           padding:5px 14px;border-radius:20px;cursor:pointer;font-size:11px;
           transition:all .15s ease}
.level-btn:hover{background:#30363d;color:#e6edf3}
.level-btn.active{background:#1f6feb;color:#fff;border-color:#388bfd}
.level-btn .cnt{font-size:9px;opacity:.7;margin-left:4px}
.level-btn.disabled{opacity:.35;cursor:default;pointer-events:none}

footer{background:#161b22;border-top:1px solid #30363d;
       padding:8px 20px;display:flex;gap:16px;font-size:11px;color:#8b949e}
.leg{display:flex;align-items:center;gap:4px}
button{background:#21262d;color:#c9d1d9;border:1px solid #30363d;
       padding:5px 12px;border-radius:5px;cursor:pointer;font-size:11px}
button:hover{background:#30363d}
#loading{position:absolute;top:50%;left:50%;
         transform:translate(-50%,-50%);color:#8b949e;font-size:14px}

/* ── Level-specific shapes via overlay ── */
.level-badge{position:absolute;top:8px;left:50%;transform:translateX(-50%);
             font-size:10px;color:#8b949e;background:#161b22;padding:2px 8px;
             border-radius:9px;border:1px solid #30363d;pointer-events:none;
             z-index:10}
</style>
</head>
<body>

<header>
  <h1>Rich Learning — Graph Explorer</h1>
  <div class="level-bar">
    <label>Level:</label>
    <div id="levelBtns"></div>
  </div>
  <button class="prop-toggle active" id="propsBtn" onclick="toggleProps()">Properties</button>
  <button onclick="reLayout()">Reset Layout</button>
  <button onclick="cy&&cy.fit(40)">Fit</button>
  <div class="stats">
    <span><span class="dot" style="background:#58a6ff"></span>
          Nodes: <b id="sL">&mdash;</b></span>
    <span><span class="dot" style="background:#3fb950"></span>
          Edges: <b id="sT">&mdash;</b></span>
    <span><span class="dot" style="background:#f0db4f"></span>
          Frontiers: <b id="sF">&mdash;</b></span>
    <span id="sLevel" style="color:#bc8cff"></span>
  </div>
</header>

<div id="main">
  <div id="props" class="side-panel open">
    <h2>Graph Overview</h2>
    <div id="propOverview"></div>
    <h2>Hierarchy Levels</h2>
    <div id="propLevels"></div>
    <h2 id="propCurTitle">Current Level</h2>
    <div id="propCurrent"></div>
    <h2>Node Statistics</h2>
    <div id="propNodes"></div>
    <h2>Edge Statistics</h2>
    <div id="propEdges"></div>
  </div>
  <div id="cy"><div id="loading">Loading graph&hellip;</div></div>
  <div id="panel" class="side-panel"><h2 id="ph">Details</h2><div id="pd"></div></div>
</div>

<footer>
  <div class="leg"><span class="dot" style="background:#58a6ff"></span>Level 0 — Base landmark</div>
  <div class="leg"><span class="dot" style="background:#bc8cff"></span>Meta-1 — Scout cluster</div>
  <div class="leg"><span class="dot" style="background:#f78166"></span>Meta-2 — Region</div>
  <div class="leg"><span class="dot" style="background:#f0db4f"></span>Meta-3 — Domain</div>
  <div class="leg"><span class="dot" style="background:#3fb950"></span>+reward</div>
  <div class="leg"><span class="dot" style="background:#da3633"></span>-reward</div>
</footer>

<script>
/* ── Palette per hierarchy level ─────────────────── */
const LEVEL_PALETTE = {
  0: ['#58a6ff','#3fb950','#e74c3c','#f0db4f','#39d2c0','#79c0ff','#56d364','#ff7b72','#d2a8ff','#ffa657'],
  1: ['#bc8cff','#a371f7','#8957e5','#6e40c9','#553098','#bc8cff','#a371f7','#8957e5','#6e40c9','#553098'],
  2: ['#f78166','#ffa657','#d29922','#e3b341','#f78166','#ffa657','#d29922','#e3b341','#f78166','#ffa657'],
  3: ['#f0db4f','#e3b341','#d29922','#bb8009','#f0db4f','#e3b341','#d29922','#bb8009','#f0db4f','#e3b341']
};
const LEVEL_SHAPES = {0:'ellipse',1:'round-diamond',2:'round-hexagon',3:'round-octagon'};
const LEVEL_NAMES  = {0:'Base',1:'Meta-1 (Scout)',2:'Meta-2 (Region)',3:'Meta-3 (Domain)'};

let cy, currentLevel = 0, levelCounts = {};

function col(clusterId, level) {
  var p = LEVEL_PALETTE[level] || LEVEL_PALETTE[0];
  return p[Math.abs(clusterId) % p.length];
}

/* ── Level buttons ────────────────────────────────── */
function renderLevelBtns() {
  var c = document.getElementById('levelBtns');
  c.innerHTML = '';
  for (var lv = 0; lv <= 3; lv++) {
    var btn = document.createElement('span');
    btn.className = 'level-btn' + (lv === currentLevel ? ' active' : '');
    var cnt = levelCounts[lv] || 0;
    if (cnt === 0 && lv > 0) btn.className += ' disabled';
    btn.innerHTML = (LEVEL_NAMES[lv] || 'L'+lv) +
      '<span class="cnt">(' + cnt + ')</span>';
    btn.dataset.lv = lv;
    btn.onclick = function() {
      var l = parseInt(this.dataset.lv);
      if ((levelCounts[l]||0) === 0 && l > 0) return;
      currentLevel = l;
      load();
    };
    c.appendChild(btn);
  }
}

/* ── Layout ───────────────────────────────────────── */
function reLayout() {
  if (!cy) return;
  var spacing = currentLevel === 0 ? 110 : 140 + currentLevel * 30;
  cy.layout({name:'cose',animate:true,animationDuration:800,
    nodeRepulsion:function(){return 6000 + currentLevel * 4000},
    idealEdgeLength:function(){return spacing},
    gravity:.25,numIter:500,padding:40}).run();
}

/* ── Transform API data → Cytoscape elements ─────── */
function build(d) {
  var lv = d.currentLevel || 0;
  var mv = Math.max(1, ...d.nodes.map(function(n){return n.visitCount}));
  var mt = Math.max(1, ...d.edges.map(function(e){return e.transitionCount}), 1);

  // Size scaling per level (meta nodes are bigger)
  var baseSize = [22, 40, 60, 80][lv] || 22;
  var sizeRange = [38, 30, 25, 20][lv] || 38;

  var nodes = d.nodes.map(function(n) {
    return {data:{
      id: n.id,
      label: n.id.length > 20 ? n.id.slice(0,20)+'\u2026' : n.id,
      vc: n.visitCount, val: n.valueEstimate, nov: n.noveltyScore,
      cid: n.clusterId, fr: n.isFrontier, hl: n.hierarchyLevel,
      cc: n.childCount || 0,
      ct: n.createdTimestep, lv: n.lastVisitedTimestep,
      sz: baseSize + (mv > 0 ? n.visitCount / mv * sizeRange : 0),
      clr: col(n.clusterId, lv),
      bw: n.isFrontier ? 3 : 0,
      bc: n.isFrontier ? '#f0db4f' : 'transparent',
      shape: LEVEL_SHAPES[lv] || 'ellipse'
    }};
  });

  var edges = d.edges.map(function(e, i) {
    return {data:{
      id: 'e'+i, source: e.source, target: e.target,
      act: e.action, rew: e.reward, cnt: e.transitionCount,
      sr: e.successRate,
      w: 1 + (mt > 0 ? e.transitionCount / mt * 4 : 0),
      clr: e.reward >= 0 ? '#238636' : '#da3633'
    }};
  });

  return nodes.concat(edges);
}

/* ── Cytoscape init ───────────────────────────────── */
function init(els, lv) {
  if (cy) cy.destroy();
  document.getElementById('loading').style.display = 'none';

  var spacing = lv === 0 ? 110 : 140 + lv * 30;

  cy = cytoscape({
    container: document.getElementById('cy'),
    elements: els,
    style: [
      {selector:'node', style:{
        'label': 'data(label)',
        'width': 'data(sz)', 'height': 'data(sz)',
        'shape': 'data(shape)',
        'background-color': 'data(clr)',
        'border-width': 'data(bw)', 'border-color': 'data(bc)',
        'color': '#8b949e', 'font-size': lv === 0 ? '9px' : '11px',
        'text-valign': 'bottom', 'text-margin-y': 6,
        'text-outline-color': '#0d1117', 'text-outline-width': 2,
        'min-zoomed-font-size': 7
      }},
      {selector:'node:selected', style:{
        'border-width': 3, 'border-color': '#fff', 'color': '#e6edf3'
      }},
      {selector:'edge', style:{
        'width': 'data(w)', 'line-color': 'data(clr)',
        'target-arrow-color': 'data(clr)', 'target-arrow-shape': 'triangle',
        'curve-style': 'bezier', 'opacity': .5, 'arrow-scale': .8
      }},
      {selector:'edge:selected', style:{
        'opacity': 1, 'width': 3,
        'line-color': '#58a6ff', 'target-arrow-color': '#58a6ff'
      }}
    ],
    layout: {name:'cose', animate:true, animationDuration:1000,
      nodeRepulsion: function(){ return 6000 + lv * 4000 },
      idealEdgeLength: function(){ return spacing },
      gravity:.25, numIter:500, padding:40},
    minZoom:.1, maxZoom:5, wheelSensitivity:.3
  });

  /* click handlers */
  cy.on('tap', 'node', function(evt) {
    var d = evt.target.data();
    var rows = [
      ['Level', (LEVEL_NAMES[d.hl] || 'Level '+d.hl)],
      ['Visits', d.vc],
      ['Value Estimate', d.val != null ? d.val.toFixed(4) : '\u2014'],
      ['Novelty Score', d.nov != null ? d.nov.toFixed(4) : '\u2014'],
      ['Cluster ID', d.cid],
      ['Frontier', d.fr ? 'Yes \u25c6' : 'No']
    ];
    if (d.cc > 0)
      rows.push(['Children', d.cc + ' nodes']);
    rows.push(['Created', 'Step ' + d.ct]);
    rows.push(['Last Visited', 'Step ' + d.lv]);
    show('Node: ' + d.id, rows);
  });

  cy.on('tap', 'edge', function(evt) {
    var d = evt.target.data();
    show('Edge', [
      ['From', d.source], ['To', d.target],
      ['Action', d.act],
      ['Reward', d.rew != null ? d.rew.toFixed(4) : '\u2014'],
      ['Traversals', d.cnt],
      ['Success Rate', (d.sr * 100).toFixed(1) + '%']
    ]);
  });

  cy.on('tap', function(evt) {
    if (evt.target === cy)
      document.getElementById('panel').classList.remove('open');
  });
}

/* ── Properties panel (left) ───────────────────────── */
function toggleProps() {
  var p = document.getElementById('props');
  var btn = document.getElementById('propsBtn');
  p.classList.toggle('open');
  btn.classList.toggle('active');
}

function fmt(n, dec) {
  if (n == null) return '\u2014';
  if (Number.isInteger(n)) return n.toLocaleString();
  return Number(n).toFixed(dec != null ? dec : 4);
}

function propsHtml(rows) {
  return rows.map(function(r) {
    if (r === '---') return '<div class="section-sep"></div>';
    return '<div class="row"><span class="k">' + r[0] +
           '</span><span class="v">' + r[1] + '</span></div>';
  }).join('');
}

function renderProps(d) {
  var a = d.aggregate || {};
  var s = d.stats || {};

  // Graph overview
  document.getElementById('propOverview').innerHTML = propsHtml([
    ['Total Landmarks', fmt(s.landmarks)],
    ['Total Transitions', fmt(s.transitions)],
    ['Hierarchy Depth', Object.keys(d.levelCounts||{}).length + ' levels']
  ]);

  // Hierarchy levels
  var lvRows = [];
  for (var lv = 0; lv <= 3; lv++) {
    var cnt = (d.levelCounts||{})[lv];
    if (cnt != null) lvRows.push([(LEVEL_NAMES[lv]||'Level '+lv), cnt + ' nodes']);
  }
  document.getElementById('propLevels').innerHTML = propsHtml(lvRows);

  // Current level header
  document.getElementById('propCurTitle').textContent =
    (d.levelName || LEVEL_NAMES[d.currentLevel] || 'Level') + ' \u2014 Stats';

  document.getElementById('propCurrent').innerHTML = propsHtml([
    ['Nodes', fmt(a.nodeCount)],
    ['Edges', fmt(a.edgeCount)],
    ['Frontiers', fmt(a.frontierCount)],
    ['Clusters', fmt(a.clusterCount)],
    ['Cluster IDs', (a.clusterIds||[]).join(', ') || '\u2014'],
    ['Total Children', fmt(a.totalChildren)]
  ]);

  // Node statistics
  document.getElementById('propNodes').innerHTML = propsHtml([
    ['Visit Sum', fmt(a.visitSum)],
    ['Visit Avg', fmt(a.visitAvg, 1)],
    ['Visit Min', fmt(a.visitMin)],
    ['Visit Max', fmt(a.visitMax)],
    '---',
    ['Value Avg', fmt(a.valueAvg)],
    ['Value Min', fmt(a.valueMin)],
    ['Value Max', fmt(a.valueMax)],
    '---',
    ['Novelty Avg', fmt(a.noveltyAvg)],
    ['Novelty Min', fmt(a.noveltyMin)],
    ['Novelty Max', fmt(a.noveltyMax)],
    '---',
    ['Timestep Range', fmt(a.timestepMin) + ' \u2013 ' + fmt(a.timestepMax)]
  ]);

  // Edge statistics
  document.getElementById('propEdges').innerHTML = propsHtml([
    ['Total Traversals', fmt(a.edgeTotalTraversals)],
    ['Avg Reward', fmt(a.edgeAvgReward)],
    ['Avg Success Rate', a.edgeAvgSuccessRate != null ? (a.edgeAvgSuccessRate*100).toFixed(1)+'%' : '\u2014'],
    ['Distinct Actions', fmt(a.distinctActionCount)],
    ['Actions', (a.distinctActions||[]).join(', ') || '\u2014']
  ]);
}

/* ── Detail panel (right, on click) ───────────────── */
function show(title, rows) {
  document.getElementById('ph').textContent = title;
  document.getElementById('pd').innerHTML = rows.map(function(r) {
    return '<div class="row"><span class="k">' + r[0] +
           '</span><span class="v">' + r[1] + '</span></div>';
  }).join('');
  document.getElementById('panel').classList.add('open');
}

/* ── Fetch & render ───────────────────────────────── */
async function load() {
  try {
    var r = await fetch('/api/graph?level=' + currentLevel);
    var d = await r.json();

    // Update level counts from response
    if (d.levelCounts) levelCounts = d.levelCounts;

    document.getElementById('sL').textContent = d.nodes.length;
    document.getElementById('sT').textContent = d.edges.length;
    document.getElementById('sF').textContent =
      d.nodes.filter(function(n){ return n.isFrontier }).length;
    document.getElementById('sLevel').textContent =
      d.levelName || LEVEL_NAMES[currentLevel] || '';

    renderLevelBtns();
    renderProps(d);
    init(build(d), currentLevel);
  } catch(e) {
    var el = document.getElementById('loading');
    el.textContent = 'Error loading graph: ' + e.message;
    el.style.display = 'block';
  }
}
load();
</script>
</body>
</html>
""";
}
