/* APETRIA E100 · Navegación y rediseño integral de las siete vistas.
   Datos: observaciones reales entregadas por apetria:observation.
   Esta vista no infiere, recalibra ni persiste resultados. */
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const summary = $('resumen-view');
  const detail = $('detail-view');
  const buttons = Array.from(document.querySelectorAll('.nav [data-view]'));
  const validViews = new Set(['resumen','metricas','anomalias','eventos','fuentes','alertas','configuracion']);
  if (!summary || !detail || buttons.length !== 7) return;

  const names = {resumen:'Resumen',metricas:'Métricas',anomalias:'Anomalías',eventos:'Eventos',fuentes:'Fuentes',alertas:'Alertas',configuracion:'Configuración'};
  const explanations = {
    metricas:'Análisis de la evaluación actual: comparación por detector, matriz de confusión y rendimiento disponible.',
    anomalias:'Explorá detecciones, anomalías etiquetadas y errores de clasificación del segmento reproducido.',
    eventos:'Trazabilidad de las observaciones recibidas por WebSocket, sin confundir la recepción con el tiempo del dataset.',
    fuentes:'Cinco fuentes experimentales. Una fuente por reproducción; no representan nodos de infraestructura conectados.',
    alertas:'Las alertas de la política seleccionada se distinguen de las señales individuales de cada detector.',
    configuracion:'Administrar fuentes y preferencias de esta instalación local. Los parámetros de reproducción están en Resumen.'
  };
  const MAX_RECORDS = 2000;
  const PAGE_SIZE = 20;
  const pageState = {anomalias:1,eventos:1,alertas:1};
  let current = 'resumen';
  let sourceInfo = {};
  let loadingSources = false;
  let lastError = '';
  let filterText = '';
  let filterModel = 'todos';
  let alertTab = 'politica';
  let renderPending = false;

  const newCounts = () => ({n:0,tp:0,fp:0,fn:0,tn:0});
  const state = {
    observations:[], anomalies:[], alerts:[], signals:[],
    seen:0, truthPositive:0, truthNegative:0, noTruth:0,
    ifPositive:0, secondPositive:0, policyPositive:0, signalTotal:0,
    latencyTotal:0, latencyN:0,
    evaluation:newCounts(), comparisons:{if:newCounts(),second:newCounts(),policy:newCounts()}
  };
  const esc = (value) => String(value === null || value === undefined ? '—' : value)
    .replace(/[&<>"']/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
  const text = (id) => $(id)?.textContent?.trim() || '—';
  const datasetName = (id) => ({smd:'SMD',nasa:'NASA SMAP/MSL',nab:'NAB AWS',hdfs:'HDFS',bgl:'BGL'}[id] || String(id || '').toUpperCase());
  const isHistorical = (id) => ['nab','hdfs','bgl'].includes(id);
  const selectedDataset = () => $('dataset')?.value || 'smd';
  const fmt = (value) => value === null || value === undefined || value === '' ? 'N/D' : String(value);
  const formatNumber = (value) => Number(value || 0).toLocaleString('es-AR');
  const getId = (d) => d.block_id ?? d.line_number ?? d.point_index ?? d.sequence_id ?? d.source_index ?? 'N/D';
  const isBit = (v) => v === 0 || v === 1;
  const isOne = (v) => v === 1;
  const selectOptions = (choices, selected) => choices.map(([key,label]) =>
    `<option value="${esc(key)}" ${key===selected?'selected':''}>${esc(label)}</option>`).join('');
  const secondName = (dataset) => ({bgl:'SGD supervisado',nab:'Robust (mediana/MAD)',hdfs:'LSTM Next-Event',smd:'LSTM Autoencoder',nasa:'LSTM Autoencoder'}[dataset] || 'Segundo detector');
  const secondKey = (dataset) => dataset==='bgl'?'sgd_prediction':dataset==='nab'?'robust_prediction':'lstm_prediction';
  const secondScore = (dataset) => dataset==='bgl'?'sgd_score':dataset==='nab'?'robust_score':'lstm_score';
  const secondThreshold = (dataset) => dataset==='bgl'?'sgd_threshold':dataset==='nab'?'robust_threshold':'lstm_threshold';

  function modelPredictions(d,dataset) {
    return [
      {name:'Isolation Forest',key:'if',pred:d.isolation_forest_prediction,score:d.isolation_forest_score,threshold:d.isolation_forest_threshold},
      {name:secondName(dataset),key:'second',pred:d[secondKey(dataset)],score:d[secondScore(dataset)],threshold:d[secondThreshold(dataset)]}
    ];
  }
  function effectivePrediction(d,policy,dataset) {
    return policy==='dual' ? d[secondKey(dataset)] : d.combined_prediction;
  }
  function updateCounts(c,truth,pred) {
    if(!isBit(truth)||!isBit(pred))return;
    c.n++;
    if(truth===1&&pred===1)c.tp++;
    else if(truth===0&&pred===1)c.fp++;
    else if(truth===1)c.fn++;
    else c.tn++;
  }
  function classification(truth,pred) {
    if(!isBit(truth)||!isBit(pred))return 'sin_datos';
    return truth===1?(pred===1?'tp':'fn'):(pred===1?'fp':'tn');
  }
  const classNames={tp:'Verdadero positivo',fp:'Falso positivo',fn:'Falso negativo',tn:'Verdadero negativo',sin_datos:'Sin clasificación'};
  const classShort={tp:'TP',fp:'FP',fn:'FN',tn:'TN',sin_datos:'N/D'};
  const chip=(label,variant='neutral') => `<span class="nav-chip nav-chip--${variant}">${esc(label)}</span>`;
  const classChip=(value) => chip(classShort[value]||'N/D',value);
  const predictionChip=(value) => isBit(value)?chip(value?'Detectada':'Normal',value?'positive':'quiet'):chip('N/D','neutral');
  const truthChip=(value) => isBit(value)?chip(value?'Anómala real':'Normal real',value?'positive':'quiet'):chip('Sin etiqueta','neutral');
  const selectedContext=() => `${datasetName(selectedDataset())} · ${text('selected-source').replace(/^.*? \/ /,'')} · ${text('validation-scope')}`;
  function modelScore(value){return Number.isFinite(value)?Number(value).toLocaleString('es-AR',{maximumFractionDigits:4}):'No disponible';}
  function recordUnit(d,dataset){return dataset==='hdfs'?'Sesión BlockId':dataset==='bgl'?'Línea de log':dataset==='nasa'?'Canal':dataset==='nab'?'Serie AWS':'Máquina';}

  function resetSession(){
    state.observations.length=0;state.anomalies.length=0;state.alerts.length=0;state.signals.length=0;
    state.seen=0;state.truthPositive=0;state.truthNegative=0;state.noTruth=0;
    state.ifPositive=0;state.secondPositive=0;state.policyPositive=0;state.signalTotal=0;
    state.latencyTotal=0;state.latencyN=0;state.evaluation=newCounts();
    state.comparisons={if:newCounts(),second:newCounts(),policy:newCounts()};
    for (const key of Object.keys(pageState)) pageState[key]=1;
    closeInspector();if(current!=='resumen')updateLive();
  }
  function keep(list,item){list.push(item);if(list.length>MAX_RECORDS)list.shift();}
  window.addEventListener('apetria:reset',resetSession);
  window.addEventListener('apetria:observation',(event)=>{
    const {dataset,observation:d,policy,partition}=event.detail||{};
    if(!d||!dataset)return;
    const models=modelPredictions(d,dataset);
    const effective=effectivePrediction(d,policy,dataset);
    const signals=models.filter(m=>isOne(m.pred));
    const truth=isBit(d.ground_truth)?d.ground_truth:null;
    const item={
      seq:++state.seen,dataset,id:getId(d),truth,policy,partition,decision:effective,
      combined:d.combined_prediction,classification:classification(truth,effective),
      flagged:signals.length>0||isOne(d.combined_prediction),
      models,received:new Date().toLocaleTimeString('es-AR',{hour12:false}),
      timestamp:d.timestamp??null,unit:recordUnit(d,dataset),
      machine:d.machine??d.channel??d.series??null,
      latencyNs:d.processing_latency_ns??null,
      windowReady:d.window_ready??null,
      historical:typeof d.historical_replay==='boolean'?d.historical_replay:isHistorical(dataset)
    };
    if(truth===1)state.truthPositive++;else if(truth===0)state.truthNegative++;else state.noTruth++;
    if(isOne(models[0].pred))state.ifPositive++;
    if(isOne(models[1].pred))state.secondPositive++;
    if(isOne(d.combined_prediction))state.policyPositive++;
    if(Number.isFinite(d.processing_latency_ns)) {state.latencyTotal+=d.processing_latency_ns/1e6;state.latencyN++;}
    updateCounts(state.evaluation,truth,effective);
    updateCounts(state.comparisons.if,truth,models[0].pred);
    updateCounts(state.comparisons.second,truth,models[1].pred);
    updateCounts(state.comparisons.policy,truth,d.combined_prediction);
    keep(state.observations,item);
    if(item.flagged||truth===1)keep(state.anomalies,item);
    if(isOne(d.combined_prediction))keep(state.alerts,item);
    for(const model of signals){keep(state.signals,{...item,signalModel:model.name});state.signalTotal++;}
    if(['metricas','anomalias','eventos','alertas'].includes(current))scheduleRender();
  });

  function scheduleRender(){if(renderPending)return;renderPending=true;requestAnimationFrame(()=>{renderPending=false;updateLive();});}
  function metricCard(label,value,caption,accent='blue'){
    return `<article class="nav-kpi nav-kpi--${accent}"><span class="nav-kpi-label">${esc(label)}</span>
      <strong class="nav-kpi-value">${esc(value)}</strong><span class="nav-kpi-caption">${esc(caption)}</span></article>`;
  }
  function section(title,desc,body,className=''){
    return `<section class="nav-panel ${className}"><div class="nav-panel-heading"><div><h3>${esc(title)}</h3>${desc?`<p>${esc(desc)}</p>`:''}</div></div>${body}</section>`;
  }
  function note(body){return `<p class="nav-explainer">${esc(body)}</p>`;}
  function percent(value,den){return den>0?(100*value/den).toFixed(2)+' %':'N/D';}
  function rate(value,den){return den>0?value/den:null;}
  function scoreStats(c){const p=rate(c.tp,c.tp+c.fp),r=rate(c.tp,c.tp+c.fn);return {
    precision:p,recall:r,f1:(p!==null&&r!==null&&p+r>0)?2*p*r/(p+r):null,
    fpr:rate(c.fp,c.fp+c.tn)
  };}
  const displayPercent=(value) => value===null?'N/D':(100*value).toFixed(2)+' %';
  function scoreBar(value,variant){const pct=value===null?0:Math.round(value*100);return `<div class="nav-meter" role="meter" aria-valuemin="0" aria-valuemax="100" ${value!==null?`aria-valuenow="${pct}"`:''} aria-label="${value===null?'Sin datos':pct+' por ciento'}"><span class="nav-meter-fill nav-meter-fill--${variant}" style="width:${pct}%"></span></div>`;}
  function comparisonRows(){
    const dataset=selectedDataset();const policy=$('policy')?.value||'and';
    return [
      {name:'Isolation Forest',desc:'Detector individual',c:state.comparisons.if,variant:'if'},
      {name:secondName(dataset),desc:'Segundo detector',c:state.comparisons.second,variant:'second'},
      {name:policy==='dual'?'Política DUAL (sin combinación)':'Política '+policy.toUpperCase(),desc:policy==='dual'?'No emite predicción combinada':'Decisión seleccionada',c:policy==='dual'?newCounts():state.comparisons.policy,variant:'policy'}
    ];
  }
  function modelComparison(){
    const rows=comparisonRows();
    return `<div class="nav-comparison"><div class="nav-comparison-head"><span>Detector / política</span><span>Evaluadas</span><span>Precision</span><span>Recall</span><span>F1-score</span><span>FPR</span></div>
      ${rows.map(row=>{const m=scoreStats(row.c);return `<div class="nav-comparison-row"><div class="nav-comparison-model"><span class="nav-model-dot nav-model-dot--${row.variant}"></span><div><strong>${esc(row.name)}</strong><small>${esc(row.desc)}</small></div></div><strong class="nav-numeric">${row.c.n?formatNumber(row.c.n):'N/D'}</strong>
        <span class="nav-numeric">${displayPercent(m.precision)}</span><span class="nav-numeric">${displayPercent(m.recall)}</span><div class="nav-score-cell"><strong>${displayPercent(m.f1)}</strong>${scoreBar(m.f1,row.variant)}</div><span class="nav-numeric">${displayPercent(m.fpr)}</span></div>`;}).join('')}</div>`;
  }
  function confusionMatrix(){const c=state.evaluation;return `<div class="nav-confusion">
    <div class="nav-confusion-cell nav-confusion-cell--tp"><span>Verdaderos positivos</span><strong>${c.n?formatNumber(c.tp):'—'}</strong><small>Pred. 1 · Real 1</small></div>
    <div class="nav-confusion-cell nav-confusion-cell--fp"><span>Falsos positivos</span><strong>${c.n?formatNumber(c.fp):'—'}</strong><small>Pred. 1 · Real 0</small></div>
    <div class="nav-confusion-cell nav-confusion-cell--fn"><span>Falsos negativos</span><strong>${c.n?formatNumber(c.fn):'—'}</strong><small>Pred. 0 · Real 1</small></div>
    <div class="nav-confusion-cell nav-confusion-cell--tn"><span>Verdaderos negativos</span><strong>${c.n?formatNumber(c.tn):'—'}</strong><small>Pred. 0 · Real 0</small></div></div>`;}
  function contextStrip(){const d=selectedDataset();const policy=($('policy')?.value||'').toUpperCase();return `<div class="nav-context"><div><span>FUENTE ACTIVA</span><strong>${esc(datasetName(d))}</strong></div><div><span>PARTICIÓN</span><strong>${esc(text('active-partition'))}</strong></div><div><span>POLÍTICA</span><strong>${esc(policy||'—')}</strong></div><div><span>ALCANCE</span><strong>Segmento reproducido</strong></div></div>`;}
  function metricsView(){
    const c=state.evaluation;const metrics=scoreStats(c),dataset=selectedDataset();
    const mainCards=[['Precision',displayPercent(metrics.precision),'TP / (TP + FP)','blue'],
      ['Recall',displayPercent(metrics.recall),'TP / (TP + FN)','green'],
      ['F1-score',displayPercent(metrics.f1),'Media armónica','purple'],
      ['Tasa de falsos positivos',displayPercent(metrics.fpr),'FP / (FP + TN)','amber']];
    const withoutDecision = Math.max(0, state.seen - c.n);
    const availabilityNote = dataset==='nasa' && ['and','or'].includes($('policy')?.value || '') && withoutDecision>0
      ? `<div class="nav-evaluation-note" role="note"><strong>${formatNumber(c.n)} de ${formatNumber(state.seen)} observaciones con decisión evaluable</strong><span>${formatNumber(withoutDecision)} observaciones no tienen una decisión de política evaluable al inicio del segmento porque todavía no se completó la ventana temporal LSTM. Isolation Forest sí puede emitir señales individuales en esas posiciones; por eso sus detecciones pueden superar las alertas finales OR/AND.</span></div>`
      : '';
    return contextStrip()+availabilityNote+`<div class="nav-kpi-grid">${mainCards.map(args=>metricCard(...args)).join('')}</div>
      <div class="nav-two-col">
        ${section('Matriz de confusión','Política de decisión frente a las etiquetas del segmento',confusionMatrix()+`<div class="nav-panel-foot">${formatNumber(c.n)} observaciones etiquetadas con decisión evaluable de ${formatNumber(state.seen)} recibidas.</div>`)}
        ${section('Resumen de predicciones','Conteos de la reproducción actual',`<div class="nav-stat-list">
          <div><span>Observaciones recibidas</span><strong>${formatNumber(state.seen)}</strong></div>
          <div><span>Detecciones IF</span><strong>${formatNumber(state.ifPositive)}</strong></div>
          <div><span>Detecciones ${esc(secondName(dataset))}</span><strong>${formatNumber(state.secondPositive)}</strong></div>
          <div><span>Alertas de política seleccionada</span><strong>${formatNumber(state.policyPositive)}</strong></div>
          <div><span>Etiquetas anómalas</span><strong>${formatNumber(state.truthPositive)}</strong></div>
          <div><span>Etiquetas normales</span><strong>${formatNumber(state.truthNegative)}</strong></div>
        </div>`)}
      </div>
      ${section('Comparación entre detectores','Cada detector se evalúa sobre las observaciones con predicción y etiqueta disponibles; la cantidad evaluada puede variar por ventanas temporales. N/D indica un denominador no disponible.',modelComparison())}
      ${section('Rendimiento y alcance','Información reportada por el backend para este tipo de reproducción',`<div class="nav-performance-grid">
        <div><span>Latencia media registrada</span><strong>${esc(text('avg-lat'))}</strong><small>${isHistorical(dataset)?'No equivale a inferencia en línea':'Medida disponible por observación'}</small></div>
        <div><span>Ventanas temporales</span><strong>${isHistorical(dataset)?'No aplica':esc(text('windows'))}</strong><small>${isHistorical(dataset)?'Predicciones históricas':'Contexto LSTM del segmento'}</small></div>
        <div><span>Transmisión</span><strong>${esc(text('stream-status'))}</strong><small>Estado de la reproducción WebSocket</small></div>
      </div>`)+note('Estas métricas corresponden solamente a la reproducción actual. No representan la evaluación global del dataset ni el throughput de inferencia. En DUAL no hay política combinada; las métricas principales siguen el segundo detector, como en Resumen.')}`;
  }

  function visibleStatus(item){return classChip(item.classification);}
  function matches(item){
    const query=filterText.toLocaleLowerCase('es-AR');
    const searchable=[item.id,item.dataset,item.policy,item.unit,item.machine,item.timestamp,item.classification,
      item.signalModel,...item.models.map(m=>m.name)].join(' ').toLocaleLowerCase('es-AR');
    if(query&&!searchable.includes(query))return false;
    if(filterModel==='todos')return true;
    if(['tp','fp','fn','tn','sin_datos'].includes(filterModel))return item.classification===filterModel;
    if(filterModel==='real_anomala')return item.truth===1;
    if(filterModel==='real_normal')return item.truth===0;
    if(filterModel==='deteccion')return item.flagged;
    if(filterModel==='politica')return isOne(item.combined);
    if(filterModel==='if')return isOne(item.models[0]?.pred);
    if(filterModel==='second')return isOne(item.models[1]?.pred);
    return true;
  }
  function filterChoices(kind){
    if(kind==='anomalias')return [['todos','Todas'],['tp','Verdaderos positivos (TP)'],['fp','Falsos positivos (FP)'],['fn','Falsos negativos (FN)'],['real_anomala','Anomalías reales'],['sin_datos','Sin etiqueta / decisión']];
    if(kind==='eventos')return [['todos','Todos'],['real_anomala','Etiqueta: anomalía'],['real_normal','Etiqueta: normal'],['deteccion','Señal de un detector'],['politica','Política positiva'],['sin_datos','Sin clasificación']];
    return [['todos','Todos'],['tp','Verdaderos positivos'],['fp','Falsos positivos'],['sin_datos','No etiquetados']];
  }
  function filteredControls(kind){return `<div class="nav-filterbar"><div class="nav-search"><span class="nav-search-icon" aria-hidden="true">⌕</span>
      <label for="nav-filter" class="nav-sr-only">Buscar observaciones</label><input id="nav-filter" type="search" placeholder="Buscar identificador, modelo o unidad..." value="${esc(filterText)}" autocomplete="off"></div>
      <label class="nav-select-label" for="nav-model-filter">Filtrar por <select id="nav-model-filter">${selectOptions(filterChoices(kind),filterModel)}</select></label>
      <span class="nav-filter-hint">${PAGE_SIZE} por página · máximo ${MAX_RECORDS} en memoria</span></div>`;}
  function paginate(items,kind){
    const ordered=items.slice().reverse();
    const total=ordered.length;
    const last=Math.max(1,Math.ceil(total/PAGE_SIZE));
    const page=Math.min(Math.max(1,pageState[kind]||1),last);
    pageState[kind]=page;
    const start=(page-1)*PAGE_SIZE;
    return {data:ordered.slice(start,start+PAGE_SIZE),total,page,pages:last,start:start+1,end:Math.min(start+PAGE_SIZE,total)};
  }
  function pagination(kind,p){
    if(p.total<=PAGE_SIZE)return '';
    return `<nav class="nav-pagination" aria-label="Páginas de ${esc(kind)}">
      <span class="nav-pagination-label">Página ${p.page} de ${p.pages}</span>
      <div class="nav-pagination-actions">
      <button type="button" data-page-kind="${esc(kind)}" data-page-number="1" ${p.page===1?'disabled':''} aria-label="Primera página">«</button>
      <button type="button" data-page-kind="${esc(kind)}" data-page-number="${p.page-1}" ${p.page===1?'disabled':''}>Anterior</button>
      <button type="button" data-page-kind="${esc(kind)}" data-page-number="${p.page+1}" ${p.page===p.pages?'disabled':''}>Siguiente</button>
      <button type="button" data-page-kind="${esc(kind)}" data-page-number="${p.pages}" ${p.page===p.pages?'disabled':''} aria-label="Última página">»</button>
      </div></nav>`;
  }
  function recordTable(headers,items,renderCells,emptyMessage,variant='default'){
    if(!items.length)return `<div class="nav-empty"><span class="nav-empty-icon">⌕</span><strong>No hay registros para esta selección</strong><p>${esc(emptyMessage)}</p></div>`;
    return `<div class="nav-table-wrap"><table class="nav-table nav-table--${variant}"><thead><tr>${headers.map(h=>`<th scope="col">${esc(h)}</th>`).join('')}</tr></thead>
      <tbody>${items.map(item=>`<tr>${renderCells(item).map(cell=>`<td>${cell}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
  }
  function recordLink(item){return `<button type="button" class="nav-record-link" data-record="${item.seq}" aria-label="Ver detalle de observación ${esc(item.id)}">${esc(item.id)}</button>`;}
  function recordIdentity(item){return `<div class="nav-record-identity">${recordLink(item)}<small>${esc(datasetName(item.dataset))}</small></div>`;}
  function resultLabel(item){return `<span class="nav-result">${classChip(item.classification)}<span>${esc(classNames[item.classification]||'Sin clasificación')}</span></span>`;}
  function recordSubtitle(kind,p){return `<div class="nav-table-summary"><span><strong>${p.total?`${formatNumber(p.start)}–${formatNumber(p.end)}`:'0'}</strong> de ${formatNumber(p.total)} registros coincidentes</span><span>${formatNumber(state.seen)} observaciones de la reproducción actual</span><span>Conservación en memoria: ${MAX_RECORDS} por colección</span></div>`;}
  function anomalyView(){
    const p=paginate(state.anomalies.filter(matches),'anomalias');
    const c=state.evaluation;
    return contextStrip()+`<div class="nav-kpi-grid">${[
      ['Anomalías reales',formatNumber(state.truthPositive),'Etiquetas 1 del segmento','green'],
      ['Detecciones positivas',formatNumber(c.tp+c.fp),'Decisión evaluada','blue'],
      ['Falsos positivos',formatNumber(c.fp),'Predicción 1 · Real 0','amber'],
      ['Falsos negativos',formatNumber(c.fn),'Predicción 0 · Real 1','red']
    ].map(args=>metricCard(...args)).join('')}</div>
    ${section('Explorador de anomalías','Observaciones con alguna señal individual o etiqueta real anómala. Incluye los falsos negativos de la política.',
      recordSubtitle('anomalias',p)+recordTable(['Observación / Fuente','Etiqueta real','Decisión de política','Resultado de clasificación'],p.data,item=>[
        recordIdentity(item),truthChip(item.truth),predictionChip(item.decision),resultLabel(item)
      ],'Probá otra clasificación o iniciá una reproducción en Resumen.','compact')+pagination('anomalias',p)+note('TP, FP y FN se calculan frente a la decisión evaluada de la política seleccionada; en DUAL se evalúa el segundo detector. Seleccioná un identificador para consultar la hora de recepción y las predicciones.'),'nav-panel--table')}`;
  }
  function timeline(){
    const recent=state.observations.slice(-60);
    if(!recent.length)return `<div class="nav-empty-small">La secuencia aparecerá al reproducir una fuente.</div>`;
    const lanes=[
      {name:'Etiqueta real',type:'truth',desc:'Anomalía / normal del dataset',get:item=>item.truth===1?'truth-anomaly':item.truth===0?'truth-normal':'unknown'},
      {name:'Señal de detector',type:'signal',desc:'Al menos un detector positivo',get:item=>item.models.some(model=>isOne(model.pred))?'signal-positive':'signal-quiet'},
      {name:'Decisión evaluada',type:'decision',desc:'Política seleccionada',get:item=>item.decision===1?'decision-positive':item.decision===0?'decision-quiet':'unknown'}
    ];
    const count=recent.length;
    return `<div class="nav-sequence" role="group" aria-label="Comparación por índice de las últimas ${count} observaciones">
      <div class="nav-sequence-lanes">${lanes.map(lane=>`<div class="nav-sequence-row">
        <div class="nav-sequence-label"><strong>${lane.name}</strong><small>${lane.desc}</small></div>
        <div class="nav-sequence-track" style="--nav-sequence-count:${count}">${recent.map(item=>`<button type="button" class="nav-sequence-tick nav-sequence-tick--${lane.get(item)}" data-record="${item.seq}" title="Observación ${esc(item.id)} · ${esc(lane.name)} · ${esc(classNames[item.classification]||'Sin clasificación')}" aria-label="Ver observación ${esc(item.id)} de ${esc(lane.name)}"></button>`).join('')}</div>
      </div>`).join('')}</div>
      <div class="nav-sequence-axis"><span>Índice inicial visible: <strong>${esc(recent[0].id)}</strong></span><span>Índice final visible: <strong>${esc(recent[recent.length-1].id)}</strong></span></div>
    </div>`;
  }
  function eventsView(){
    const p=paginate(state.observations.filter(matches),'eventos');
    return contextStrip()+`<div class="nav-three-stats">
      ${metricCard('Eventos recibidos',formatNumber(state.seen),'Mensajes de observación','blue')}
      ${metricCard('Con etiqueta real',formatNumber(state.truthPositive+state.truthNegative),'Observaciones evaluables','green')}
      ${metricCard('Sin etiqueta real',formatNumber(state.noTruth),'Sin verdad de referencia','amber')}
    </div>
    ${section('Secuencia de observaciones','Últimos 60 registros conservados en orden de recepción. Seleccioná una marca para consultar su detalle. Los tres carriles muestran datos diferentes, no una medición de tiempo continuo.',timeline()+`<div class="nav-sequence-legend"><span><i class="nav-legend-dot nav-legend-dot--anomaly"></i>Anomalía real</span><span><i class="nav-legend-dot nav-legend-dot--normal"></i>Normal real</span><span><i class="nav-legend-dot nav-legend-dot--signal"></i>Detector positivo</span><span><i class="nav-legend-dot nav-legend-dot--decision"></i>Política positiva</span><span><i class="nav-legend-dot nav-legend-dot--unknown"></i>No disponible</span></div>`)}
    ${section('Registro de eventos','Trazabilidad de observaciones recibidas durante la reproducción.',
      recordSubtitle('eventos',p)+recordTable(['Observación / Fuente','Etiqueta real','Decisión evaluada','Timestamp original','Recepción local'],p.data,item=>[
        recordIdentity(item),truthChip(item.truth),predictionChip(item.decision),`<span class="nav-mono nav-timestamp" title="${esc(item.timestamp??'Sin timestamp en la fuente')}">${esc(item.timestamp??'No provisto')}</span>`,`<span class="nav-mono">${esc(item.received)}</span>`
      ],'No hay observaciones para los filtros elegidos.','events')+pagination('eventos',p)+note('El timestamp original solo aparece cuando el adaptador lo proporciona. La hora local indica llegada por WebSocket al navegador; no es una medida de latencia de inferencia.'),'nav-panel--table')}`;
  }
  function alertsView(){const dataset=selectedDataset();const policy=($('policy')?.value||'and').toUpperCase();const dual=policy==='DUAL';
    const source=alertTab==='politica'?state.alerts:state.signals;
    const p=paginate(source.filter(matches),'alertas');
    const data=p.data;
    return contextStrip()+`<div class="nav-kpi-grid">${[
      ['Alertas de política',formatNumber(state.policyPositive),dual?'DUAL no combina modelos':`Predicción ${policy} positiva`,'blue'],
      ['Señales individuales',formatNumber(state.signalTotal),'Suma de señales de los detectores','purple'],
      ['Detecciones IF',formatNumber(state.ifPositive),'Isolation Forest positivo','green'],
      ['Señales '+secondName(dataset),formatNumber(state.secondPositive),'Segundo detector positivo','amber']
    ].map(args=>metricCard(...args)).join('')}</div>
    ${section('Centro de detecciones',dual?'DUAL permite comparar modelos individualmente, sin alerta híbrida final.':'Se registran las decisiones positivas de la política seleccionada, ya sea individual (IF, LSTM, SGD) o combinada (AND, OR).',
      `<div class="nav-tabs" role="group" aria-label="Tipo de registros de alerta"><button type="button" data-alert-tab="politica" class="${alertTab==='politica'?'active':''}" aria-pressed="${alertTab==='politica'}">Alertas de política <span>${formatNumber(state.policyPositive)}</span></button>
       <button type="button" data-alert-tab="individual" class="${alertTab==='individual'?'active':''}" aria-pressed="${alertTab==='individual'}">Señales por detector <span>${formatNumber(state.signalTotal)}</span></button></div>
       ${recordSubtitle('alertas',p)}
       ${recordTable(alertTab==='politica'?['Observación','Fuente','Política','Etiqueta real','Clasificación','Recepción local']:['Observación','Fuente','Detector que señaló','Etiqueta real','Decisión de política','Recepción local'],data,item=>alertTab==='politica'?[
         recordLink(item),`<strong>${esc(datasetName(item.dataset))}</strong>`,chip((item.policy||'').toUpperCase(),'positive'),truthChip(item.truth),visibleStatus(item),`<span class="nav-mono">${esc(item.received)}</span>`
       ]:[recordLink(item),`<strong>${esc(datasetName(item.dataset))}</strong>`,`<span>${esc(item.signalModel)}</span>`,truthChip(item.truth),predictionChip(item.combined),`<span class="nav-mono">${esc(item.received)}</span>`],
         alertTab==='politica'?'La política no generó alertas positivas en este segmento. Podés consultar las señales individuales en la otra pestaña.':'No hay señales individuales que coincidan con los filtros.','alerts')}
       ${pagination('alertas',p)}
       ${note('Los registros son resultados experimentales conservados en memoria; no son incidentes abiertos ni notificaciones enviadas a servicios externos.')}`,'nav-panel--table')}`;
  }
  function recordInspector(item){if(!item)return '';
    const models=[...item.models,{name:'Política combinada',pred:item.combined,score:null,threshold:null}];
    const evidence=models.map(m=>`<div class="nav-inspector-model"><div><strong>${esc(m.name)}</strong>${predictionChip(m.pred)}</div><div><span>Score</span><strong>${esc(modelScore(m.score))}</strong></div><div><span>Umbral</span><strong>${esc(modelScore(m.threshold))}</strong></div></div>`).join('');
    const entries=[['Dataset',datasetName(item.dataset)],['Identificador',item.id],['Unidad',item.machine??item.unit],['Partición',item.partition??'No provista'],['Política',(item.policy||'').toUpperCase()],['Etiqueta real',fmt(item.truth)],['Decisión evaluada',fmt(item.decision)],['Clasificación',classNames[item.classification]||'No disponible'],['Timestamp original',item.timestamp??'No provisto'],['Hora local de recepción',item.received],['Procesamiento',item.historical?'Predicciones históricas':'Inferencia previa a la transmisión']];
    return `<div class="nav-inspector-heading"><span>EXPLORAR OBSERVACIÓN</span><button type="button" data-close-inspector aria-label="Cerrar detalle">×</button></div>
      <h3>${esc(item.id)}</h3><p>Resultado de la reproducción actual · ${esc(datasetName(item.dataset))}</p>
      <div class="nav-inspector-summary">${visibleStatus(item)}${predictionChip(item.decision)}</div>
      <div class="nav-inspector-entries">${entries.map(([a,b])=>`<div><span>${esc(a)}</span><strong>${esc(b)}</strong></div>`).join('')}</div>
      <h4>Predicciones disponibles</h4>${evidence}
      ${note('N/D significa que no existe una predicción evaluable. Los scores y umbrales solo se muestran si el backend los proporcionó.')}`;
  }
  function showInspector(seq){const item=state.observations.find(x=>x.seq===seq) || state.alerts.find(x=>x.seq===seq) || state.signals.find(x=>x.seq===seq) || state.anomalies.find(x=>x.seq===seq);const inspector=$('nav-inspector');if(!item||!inspector)return;
    inspector.innerHTML=recordInspector(item);inspector.hidden=false;inspector.setAttribute('aria-hidden','false');
    inspector.querySelector('[data-close-inspector]')?.focus();
  }
  function closeInspector(){const inspector=$('nav-inspector');if(inspector){inspector.hidden=true;inspector.innerHTML='';inspector.setAttribute('aria-hidden','true');}}
  detail.addEventListener('click',(event)=>{
    const button=event.target.closest('button');if(!button)return;
    if(button.hasAttribute('data-record')){showInspector(Number(button.dataset.record));return;}
    if(button.hasAttribute('data-close-inspector')){closeInspector();return;}
    if(button.hasAttribute('data-alert-tab')){alertTab=button.dataset.alertTab;pageState.alertas=1;updateLive();return;}
    if(button.hasAttribute('data-page-kind')){
      const kind=button.dataset.pageKind;const next=Number(button.dataset.pageNumber);
      if(!Object.hasOwn(pageState,kind)||!Number.isSafeInteger(next)||next<1)return;
      pageState[kind]=next;updateLive();
      $('detail-live')?.querySelector('.nav-panel--table')?.scrollIntoView({block:'start',behavior:'auto'});
    }
  });
  document.addEventListener('keydown',(event)=>{if(event.key==='Escape'&&!$('nav-inspector')?.hidden)closeInspector();});
  function updateLive(){const content=$('detail-live');if(!content)return;
    if(current==='metricas')content.innerHTML=metricsView();
    else if(current==='anomalias')content.innerHTML=anomalyView();
    else if(current==='eventos')content.innerHTML=eventsView();
    else if(current==='alertas')content.innerHTML=alertsView();
  }
  async function refreshSources() {
    if (loadingSources) return;
    loadingSources = true;
    lastError = '';
    try {
      const response = await fetch('/api/sources');
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      sourceInfo = Object.fromEntries((data.sources || []).map(x => [x.id,x]));
    } catch (err) { lastError = String(err.message || err); }
    finally { loadingSources = false; if (current === 'fuentes') renderSources(); }
  }
  function renderSources() {
    const target = $('detail-live');
    if (!target) return;
    if (lastError) {
      target.innerHTML = `<div class="nav-empty"><strong>No se pudo consultar las fuentes</strong><p>${esc(lastError)}</p><button type="button" class="detail-button" id="sources-retry">Reintentar</button></div>`;
      target.querySelector('#sources-retry')?.addEventListener('click',()=>void refreshSources());
      return;
    }
    const ids=['smd','nasa','nab','hdfs','bgl'];
    const loaded=ids.filter(id=>sourceInfo[id]?.status==='disponible').length;
    const cards=ids.map((id,index)=>{
      const source=sourceInfo[id];const ready=source?.status==='disponible';const selected=selectedDataset()===id;
      const units=Array.isArray(source?.machines)?source.machines:[];
      const count=id==='hdfs'?(source?.total_sessions??0):id==='bgl'?(source?.total_events??0):units.length;
      const unitLabel=id==='hdfs'?'sesiones BlockId':id==='bgl'?'eventos de log':id==='nasa'?'canales':id==='nab'?'series AWS':'máquinas';
      const explanation=isHistorical(id)?'Predicciones calculadas previamente. La consola reproduce resultados históricos sin ejecutar inferencia en línea.':'Inferencia sobre observaciones reales del dataset; los modelos se ejecutan antes de transmitir los resultados.';
      const model=id==='bgl'?'Isolation Forest + SGD':id==='nab'?'Isolation Forest + Robust':id==='hdfs'?'Isolation Forest + LSTM Next-Event':'Isolation Forest + LSTM Autoencoder';
      const unitDetail=units.length?`<details class="source-units"><summary>Explorar ${formatNumber(units.length)} ${esc(unitLabel)} disponibles</summary><div class="source-units-list">${units.map(u=>`<span title="${esc(u)}">${esc(u)}</span>`).join('')}</div></details>`:
        `<p class="source-unit-footnote">${id==='hdfs'||id==='bgl'?'Los identificadores individuales se exploran en Eventos después de reproducir.':'No se informaron unidades seleccionables en esta instalación.'}</p>`;
      return `<article class="source-catalog-card ${selected?'source-catalog-card--selected':''}" data-source-card="${id}">
        <div class="source-catalog-top"><span class="source-catalog-number">0${index+1} · ${isHistorical(id)?'HISTÓRICO':'INFERENCIA'}</span>${chip(ready?'Disponible':source?.status==='deshabilitada'?'Deshabilitada':source?'Sin artefactos':'Consultando',ready?'positive':'neutral')}</div>
        <h3>${esc(datasetName(id))}</h3><p class="source-catalog-desc">${esc(explanation)}</p>
        <div class="source-catalog-stat"><strong>${source?formatNumber(count):'—'}</strong><span>${esc(unitLabel)}</span></div>
        <div class="source-catalog-model"><span>DETECTORES</span><strong>${esc(model)}</strong></div>
        ${unitDetail}
        <button type="button" class="source-catalog-action" data-activate="${id}" ${!ready?'disabled':''}>${selected?'Fuente activa · Volver a Resumen':ready?'Seleccionar y configurar en Resumen':'No disponible para reproducir'}</button>
      </article>`;
    }).join('');
    target.innerHTML=`<section class="source-intro"><div><span class="nav-eyebrow">CATÁLOGO EXPERIMENTAL</span><h3>Cinco fuentes, una sesión activa</h3><p>Elegí un dataset y configurá su unidad, partición y política en Resumen. Estas fuentes no representan servidores conectados ni se ejecutan simultáneamente.</p></div><div class="source-intro-metrics"><strong>${loaded}/5</strong><span>fuentes disponibles</span></div></section>
    <div class="source-catalog-grid">${cards}</div>
    ${note('Los conteos y las unidades provienen de /api/sources. Deshabilitar o habilitar fuentes se administra en Configuración; no elimina datasets ni artefactos.')}`;
    target.querySelectorAll('[data-activate]').forEach(btn=>btn.addEventListener('click',()=>{
      const id=btn.dataset.activate;
      if(sourceInfo[id]?.status!=='disponible')return;
      if(selectedDataset()!==id){$('dataset').value=id;$('dataset').dispatchEvent(new Event('change',{bubbles:true}));}
      navigate('resumen');
      document.querySelector('.replay-controls-sticky')?.scrollIntoView({behavior:'smooth',block:'start'});
    }));
  }
  const sourceOptions = () => ['smd','nasa','nab','hdfs','bgl'].map(id=>[id,datasetName(id)]);
  const policyOptions = (dataset) => dataset==='bgl' ? [['sgd','SGD supervisado'],['if','Isolation Forest'],['and','AND'],['or','OR']] :
    dataset==='hdfs' ? [['lstm','LSTM Next-Event'],['if','Isolation Forest'],['and','AND'],['or','OR']] : [['and','AND'],['or','OR'],['dual','DUAL']];

  async function buildConfig() {
    const target=$('detail-live');
    if (!target) return;
    target.innerHTML='<div class="detail-empty">Cargando configuración local...</div>';
    try {
      const responses=await Promise.all([
        fetch('/api/config'),
        fetch('/api/sources')
      ]);
      if (responses.some(r=>!r.ok)) throw new Error('No se pudo leer la configuración del backend.');
      const settings=await responses[0].json();
      const sourcePayload=await responses[1].json();
      sourceInfo=Object.fromEntries((sourcePayload.sources||[]).map(item=>[item.id,item]));
      if (current!=='configuracion') return;
      const rows=sourceOptions().map(([id,name])=>{
        const source=sourceInfo[id];
        const artifacts=source?.artifacts_available??source?.status==='disponible';
        return `<label class="admin-source-row" for="admin-source-${id}">
          <span><strong>${esc(name)}</strong><small>${artifacts?'Artefactos detectados':'Artefactos no disponibles en esta instalación'}</small></span>
          <span class="admin-source-state"><input type="checkbox" id="admin-source-${id}" data-admin-source="${id}" ${settings.enabled_sources.includes(id)?'checked':''}/><span>Habilitada</span></span>
        </label>`;
      }).join('');
      const adminName=window.APETRIA_AUTH?.username||'Administrador';
      const enabledCount=settings.enabled_sources.length;
      target.innerHTML=`<div class="admin-overview"><div><span class="nav-eyebrow">CONFIGURACIÓN LOCAL</span><h3>Administración de la instancia</h3><p>Los cambios se almacenan en esta instalación. No modifican los modelos entrenados, ni los resultados históricos de la tesis.</p></div><div class="admin-overview-counter"><strong>${enabledCount}/5</strong><span>fuentes habilitadas</span></div></div>
      <div class="admin-layout">
        <div class="admin-layout-main">
          <form id="admin-settings-form">
            <section class="detail-section admin-panel"><div class="admin-panel-heading"><span class="admin-panel-icon">01</span><div><h3>Administración de fuentes</h3><p class="detail-help">Habilitá o deshabilitá las cinco fuentes. El backend restringe las reproducciones REST y WebSocket de una fuente deshabilitada.</p></div></div>
              <div class="admin-source-list">${rows}</div>
            </section>
            <section class="detail-section admin-panel"><div class="admin-panel-heading"><span class="admin-panel-icon">02</span><div><h3>Preferencias iniciales</h3><p class="detail-help">Aplican al abrir APETRIA. La unidad y el índice de cada reproducción se eligen en Resumen.</p></div></div>
              <div class="detail-form admin-preferences">
                <label>Dataset predeterminado<select id="admin-default-dataset">${selectOptions(sourceOptions(),settings.default_dataset)}</select></label>
                <label>Política predeterminada<select id="admin-default-policy"></select></label>
                <label>Observaciones predeterminadas<input id="admin-default-limit" type="number" min="30" max="10000" step="1" required value="${esc(settings.default_limit)}" /></label>
                <label>Apariencia<select id="admin-theme">${selectOptions([['dark','Tema oscuro'],['light','Tema claro']],settings.theme)}</select></label>
              </div>
              <div class="admin-form-actions"><button class="detail-button" type="submit" id="admin-save">Guardar preferencias</button><span class="admin-message" id="admin-message" role="status" aria-live="polite">Los cambios se guardan en esta instalación local.</span></div>
            </section>
          </form>
        </div>
        <aside class="admin-layout-side">
          <section class="detail-section admin-panel admin-account"><div class="admin-panel-heading"><span class="admin-panel-icon">03</span><div><h3>Cuenta y seguridad</h3><p class="detail-help">Administrador autenticado de esta instalación.</p></div></div>
            <div class="admin-avatar" aria-hidden="true">${esc(adminName.slice(0,1).toUpperCase())}</div>
            <div class="admin-user-name">${esc(adminName)}</div>
            <span class="admin-user-role">Administrador local · Cuenta única</span>
            <div class="admin-account-facts"><div><span>Autenticación</span><strong>Sesión activa</strong></div><div><span>Almacenamiento</span><strong>SQLite local</strong></div><div><span>Control de acceso</span><strong>API y WebSocket</strong></div></div>
            <button type="button" class="admin-account-action" id="admin-open-profile">Cambiar contraseña / Ver perfil</button>
            <button type="button" class="admin-account-action admin-account-action--outline" id="admin-logout">Cerrar sesión</button>
            <p class="admin-account-tip">La contraseña no se muestra ni se almacena en el navegador. La cuenta y las sesiones permanecen fuera de Git en <code>instance/auth.db</code>.</p>
          </section>
          <section class="detail-section admin-panel admin-security-note"><h3>Acerca de la reproducción</h3><p>Los datasets pueden contener etiquetas históricas. Los datos experimentales no son alertas de producción ni notificaciones enviadas a terceros.</p></section>
        </aside>
      </div>`;
      target.querySelector('#admin-open-profile')?.addEventListener('click',()=>$('profile-button')?.click());
      target.querySelector('#admin-logout')?.addEventListener('click',()=>$('logout-button')?.click());
      const enabledSources=()=>Array.from(target.querySelectorAll('[data-admin-source]:checked')).map(e=>e.dataset.adminSource);
      const defaultDataset=$('admin-default-dataset');
      const defaultPolicy=$('admin-default-policy');
      function updatePolicies(keep) {
        const options=policyOptions(defaultDataset.value);
        const previous=keep || defaultPolicy.value;
        defaultPolicy.innerHTML=selectOptions(options,options.some(([value])=>value===previous)?previous:options[0][0]);
      }
      function updateChoices() {
        const enabled=enabledSources();
        const currentValue=defaultDataset.value;
        for(const option of defaultDataset.options) option.disabled=!enabled.includes(option.value);
        if(!enabled.includes(currentValue)) defaultDataset.value=enabled[0]||'';
        updatePolicies();
        const submit=$('admin-save');
        submit.disabled=!enabled.length;
        $('admin-message').textContent=!enabled.length?'Debe quedar por lo menos una fuente habilitada.':'Los cambios se guardan en esta instalación local.';
      }
      updatePolicies(settings.default_policy);
      target.querySelectorAll('[data-admin-source]').forEach(el=>el.addEventListener('change',updateChoices));
      defaultDataset.addEventListener('change',()=>updatePolicies());
      updateChoices();
      $('admin-settings-form').addEventListener('submit', async (event)=>{
        event.preventDefault();
        const enabled=enabledSources();
        const limit=Number($('admin-default-limit').value);
        const message=$('admin-message');
        if(!enabled.length||!enabled.includes(defaultDataset.value)||!Number.isSafeInteger(limit)||limit<30||limit>10000){
          message.textContent='Revisá las fuentes habilitadas y el límite (30 a 10000).';
          message.classList.add('error');return;
        }
        const payload={enabled_sources:enabled,default_dataset:defaultDataset.value,default_policy:defaultPolicy.value,default_limit:limit,theme:$('admin-theme').value};
        $('admin-save').disabled=true;
        message.classList.remove('error');message.textContent='Guardando...';
        try {
          const response=await fetch('/api/config',{
            method:'PUT',headers:{'Content-Type':'application/json','X-CSRF-Token':window.APETRIA_AUTH?.csrf||''},body:JSON.stringify(payload)
          });
          if(!response.ok){
            let detail='HTTP '+response.status;
            try{const body=await response.json();detail=typeof body.detail==='string'?body.detail:detail;}catch(_){}
            throw new Error(detail);
          }
          const saved=await response.json();
          localStorage.setItem('apetria-theme',saved.theme);
          message.textContent='Guardado. Actualizando fuentes y preferencias...';
          // Cancela una reproducción activa y sincroniza los selectores con /api/sources.
          window.location.reload();
        }catch(error){
          message.textContent='No se pudo guardar: '+(error.message||error);
          message.classList.add('error');$('admin-save').disabled=false;
        }
      });
    }catch(error){
      if(current==='configuracion')target.innerHTML=`<div class="detail-empty">${esc(error.message||error)}. Revisá FastAPI y el endpoint /api/config.</div>`;
    }
  }

  function renderPage(){
    if(current==='resumen')return;
    const analysis=['anomalias','eventos','alertas'].includes(current);
    const eyebrow=['metricas','anomalias','eventos','alertas'].includes(current)?'LABORATORIO DE OBSERVABILIDAD':'ADMINISTRACIÓN DE LA INSTALACIÓN';
    detail.innerHTML=`<div class="detail-head nav-page-head"><div><div class="nav-eyebrow">${eyebrow}</div><h2>${esc(names[current])}</h2><p>${esc(explanations[current]||'')}</p></div>
      <button type="button" class="detail-button nav-summary-button" id="nav-go-summary">Volver a Resumen</button></div>
      ${analysis?filteredControls(current):''}
      <div id="detail-live"></div><aside id="nav-inspector" class="nav-inspector" role="dialog" aria-modal="false" aria-label="Detalle de observación" hidden aria-hidden="true"></aside>`;
    $('nav-go-summary').addEventListener('click',()=>navigate('resumen'));
    if(analysis){
      $('nav-filter').addEventListener('input',event=>{filterText=event.target.value;pageState[current]=1;updateLive();});
      $('nav-model-filter').addEventListener('change',event=>{filterModel=event.target.value;pageState[current]=1;updateLive();});
    }
    if(current==='fuentes'){renderSources();void refreshSources();}
    else if(current==='configuracion'){void buildConfig();}
    else updateLive();
  }
  function navigate(page){
    if(!validViews.has(page))return;
    current=page;const active=page==='resumen';
    summary.hidden=!active;detail.hidden=active;
    buttons.forEach(b=>{const on=b.dataset.view===page;b.classList.toggle('active',on);
      if(on)b.setAttribute('aria-current','page');else b.removeAttribute('aria-current');});
    filterText='';filterModel='todos';alertTab='politica';
    if(Object.hasOwn(pageState,page))pageState[page]=1;
    if(!active)renderPage();
    requestAnimationFrame(()=>{window.scrollTo(0,0);window.dispatchEvent(new Event('resize'));});
  }
  summary.querySelectorAll('[data-summary-view]').forEach(button=>button.addEventListener('click',()=>navigate(button.dataset.summaryView)));
  buttons.forEach(button=>button.addEventListener('click',()=>navigate(button.dataset.view)));
  navigate('resumen');
})();
