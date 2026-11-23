(() => {
'use strict';
const $ = id => document.getElementById(id);
const API = ''; // Mismo origen: frontend y backend se sirven desde FastAPI.
const appRoot = document.querySelector('.app');
$('menu-toggle').onclick = () => {
 const collapsed = appRoot.classList.toggle('sidebar-collapsed');
 $('menu-toggle').setAttribute('aria-expanded', String(!collapsed));
 $('menu-toggle').setAttribute('aria-label', collapsed ? 'Expandir menú' : 'Contraer menú');
 try { localStorage.setItem('apetria-sidebar-collapsed', String(collapsed)); } catch (_) {}
};
try { if (localStorage.getItem('apetria-sidebar-collapsed') === 'true') {appRoot.classList.add('sidebar-collapsed');$('menu-toggle').setAttribute('aria-expanded','false');} } catch (_) {}
function setStreaming(state) {
 const labels={idle:'STREAMING INACTIVO',connecting:'CONECTANDO STREAMING',active:'STREAMING ACTIVO · REPRODUCCIÓN',done:'REPRODUCCIÓN COMPLETADA',stopped:'STREAMING DETENIDO',error:'ERROR DE STREAMING'};
 const badge=$('stream-badge');
 badge.className='top-badge stream '+(state==='active'?'active':state==='done'?'done':state==='error'?'error':'idle');
 $('stream-indicator').textContent=labels[state]||labels.idle;
}

let socket = null, runId = 0, count = 0, ifCount = 0, lstmCount = 0, hybridCount = 0, windows = 0, latencySum = 0, latencyN = 0, alertRows = 0;
let uniqueAlertCount=0;
let confusion={tp:0,fp:0,fn:0,tn:0}, labeled=0, truthRuns=[], validationPolicy="and", validationPartition="X_test";
let comparison={if:{tp:0,fp:0,fn:0,tn:0,n:0},lstm:{tp:0,fp:0,fn:0,tn:0,n:0},hybrid:{tp:0,fp:0,fn:0,tn:0,n:0}};
const fmt = (x, n=3) => x == null || !Number.isFinite(Number(x)) ? '—' : Number(x).toFixed(n);
const setStatus = (message, error=false) => { $('status').textContent=message; $('status').classList.toggle('error',error); };
const chart = new Chart($('chart-metrics').getContext('2d'), {
 type:'line',data:{labels:[],datasets:[
 {label:'Isolation Forest',data:[],borderColor:'#4c9bff',pointRadius:0,tension:.2,yAxisID:'yIF'},
 {label:'Umbral IF',data:[],borderColor:'#4c9bff',borderDash:[5,5],pointRadius:0,borderWidth:1,yAxisID:'yIF'},
 {label:'LSTM Autoencoder',data:[],borderColor:'#b680ff',pointRadius:0,tension:.2,yAxisID:'yLSTM'},
 {label:'Umbral LSTM',data:[],borderColor:'#ffbf27',borderDash:[5,5],pointRadius:0,borderWidth:1,yAxisID:'yLSTM'}]},
 options:{responsive:true,maintainAspectRatio:false,animation:false,interaction:{mode:'index',intersect:false},plugins:{legend:{labels:{color:'#c1ccda'}}},scales:{x:{ticks:{color:'#9caabd',maxTicksLimit:9},grid:{color:'#203043'}},yIF:{position:'left',ticks:{color:'#83b7ff'},grid:{color:'#203043'}},yLSTM:{position:'right',ticks:{color:'#c7a7ff'},grid:{drawOnChartArea:false}}}}
});
function theme(t){document.documentElement.dataset.theme=t;localStorage.setItem('apetria-theme',t);$('theme-label').textContent=t==='light'?'Tema claro':'Tema oscuro';const light=t==='light';chart.options.plugins.legend.labels.color=light?'#475569':'#c1ccda';chart.options.scales.x.ticks.color=light?'#64748b':'#9caabd';chart.options.scales.x.grid.color=light?'#e2e8f0':'#203043';chart.options.scales.yIF.grid.color=light?'#e2e8f0':'#203043';chart.update('none');}
$('theme-toggle').onclick=()=>theme(document.documentElement.dataset.theme==='light'?'dark':'light');
try{theme(localStorage.getItem('apetria-theme')==='light'?'light':'dark')}catch(_){theme('dark')}
function clearView(){comparison={if:{tp:0,fp:0,fn:0,tn:0,n:0},lstm:{tp:0,fp:0,fn:0,tn:0,n:0},hybrid:{tp:0,fp:0,fn:0,tn:0,n:0}};$("comparison-scope").textContent="Esperando reproducción X_test";$("comparison-body").innerHTML="<tr><td colspan=\"10\" style=\"text-align:left;color:var(--muted)\">Esperando resultados de X_test.</td></tr>";uniqueAlertCount=0;confusion={tp:0,fp:0,fn:0,tn:0};labeled=0;truthRuns=[];for(const key of ["tp","fp","fn","tn","precision","recall","f1","fpr"])$("val-"+key).textContent="—";$("validation-timeline").replaceChildren();$("validation-label-count").textContent="Etiquetas: —";$("validation-scope").textContent="Esperando reproducción X_test";count=ifCount=lstmCount=hybridCount=windows=latencySum=latencyN=alertRows=0;for(const id of ['processed','ifalerts','lstmalerts','combined','windows','critical-count'])$(id).textContent='0';if(['hdfs','nab','bgl'].includes($('dataset').value))$('windows').textContent='N/D';$('avg-lat').textContent='—';$('window-status').textContent='Pendiente';$('alerts-body').replaceChildren();chart.data.labels=[];chart.data.datasets.forEach(s=>s.data=[]);chart.update('none');window.dispatchEvent(new CustomEvent('apetria:reset'));}
function addAlert(d,model,score){const row=document.createElement('tr');const isPolicy=/^(h[ií]brido|pol[ií]tica)/i.test(model);for(const value of [d.point_index ?? d.sequence_id,model,fmt(score),isPolicy?'Política':'Señal']){const cell=document.createElement('td');cell.textContent=value;row.appendChild(cell)}row.lastElementChild.style.color=isPolicy?'#5cd5c6':'#d7adff';$('alerts-body').prepend(row);if($('alerts-body').children.length>30)$('alerts-body').lastElementChild.remove();alertRows++;}
function pctOrND(num,den){return den===0?'N/D':(100*num/den).toFixed(2)+' %';}
function recordComparison(c,truth,pred){if(truth!==0&&truth!==1)return;if(pred!==0&&pred!==1)return;c.n++;if(truth===1&&pred===1)c.tp++;else if(truth===0&&pred===1)c.fp++;else if(truth===1)c.fn++;else c.tn++;}
function comparisonMetrics(c){const p=c.tp+c.fp?c.tp/(c.tp+c.fp):null;const r=c.tp+c.fn?c.tp/(c.tp+c.fn):null;const f1=p!==null&&r!==null&&p+r>0?2*p*r/(p+r):null;return [pctOrND(c.tp,c.tp+c.fp),pctOrND(c.tp,c.tp+c.fn),f1===null?'N/D':(100*f1).toFixed(2)+' %',pctOrND(c.fp,c.fp+c.tn)];}
function renderComparison(){const body=$('comparison-body');body.replaceChildren();if(validationPartition!=='X_test' && validationPartition!=='historical_test'){const tr=document.createElement('tr');const td=document.createElement('td');td.colSpan=10;td.textContent='X_val: no hay etiquetas de referencia verificadas para esta tabla.';tr.appendChild(td);body.appendChild(tr);return;}
 const isHDFSComparison = $('dataset').value === 'hdfs';
 const isBGLComparison = $('dataset').value === 'bgl';
 const secondModelName = isBGLComparison
    ? 'SGD supervisado'
    : isHDFSComparison
    ? 'LSTM Next-Event'
    : $('dataset').value === 'nab'
        ? 'Robust (mediana/MAD)'
        : 'LSTM Autoencoder';
 const selectedPolicyName = (isHDFSComparison || isBGLComparison)
    ? 'Politica seleccionada: ' + validationPolicy.toUpperCase()
    : 'H?brido ' + validationPolicy.toUpperCase();
 const rows=[
    ['Isolation Forest',comparison.if,false],
    [secondModelName,comparison.lstm,false],
    [selectedPolicyName,comparison.hybrid,true]
 ];
 for(const [name,c,hybrid] of rows){const tr=document.createElement('tr');if(hybrid&&validationPolicy!=='dual')tr.className='comparison-selected';const cells=(hybrid&&validationPolicy==='dual')?[name,'N/D','—','—','—','—','N/D','N/D','N/D','N/D']:[name,c.n,c.tp,c.fp,c.fn,c.tn,...comparisonMetrics(c)];for(const v of cells){const td=document.createElement('td');td.textContent=String(v);tr.appendChild(td)}body.appendChild(tr)}
}
function updateComparison(d){if(!['X_test','historical_test'].includes(validationPartition)||(d.ground_truth!==0&&d.ground_truth!==1))return;recordComparison(comparison.if,d.ground_truth,d.isolation_forest_prediction);recordComparison(comparison.lstm,d.ground_truth,d.lstm_prediction);recordComparison(comparison.hybrid,d.ground_truth,d.combined_prediction);if(count%10===0)renderComparison();}
function updateValidation(d){
 if(!["X_test","historical_test"].includes(validationPartition) || (d.ground_truth!==0 && d.ground_truth!==1))return;
 const pred=validationPolicy==="dual"?d.lstm_prediction:d.combined_prediction;
 if(pred!==0 && pred!==1)return;
 labeled++;
 const truth=d.ground_truth;
 if(truth===1&&pred===1)confusion.tp++;
 else if(truth===0&&pred===1)confusion.fp++;
 else if(truth===1&&pred===0)confusion.fn++;
 else confusion.tn++;
 const last=truthRuns[truthRuns.length-1];if(last&&last.truth===truth)last.count++;else truthRuns.push({truth,count:1});
 for(const key of ["tp","fp","fn","tn"])$("val-"+key).textContent=confusion[key];
 const {tp,fp,fn,tn}=confusion;
 const precision=tp+fp?tp/(tp+fp):null,recall=tp+fn?tp/(tp+fn):null;
 const f1=precision!==null&&recall!==null&&precision+recall>0?2*precision*recall/(precision+recall):null;
 const fpr=fp+tn?fp/(fp+tn):null;
 for(const [key,value] of Object.entries({precision,recall,f1,fpr}))$("val-"+key).textContent=value===null?"N/D":(100*value).toFixed(2)+" %";
 $("validation-label-count").textContent="Etiquetas evaluadas: "+labeled+" · Anómalas: "+(tp+fn)+" · Normales: "+(fp+tn);
}
function drawTruthTimeline(){const el=$("validation-timeline");el.replaceChildren();for(const run of truthRuns){const bar=document.createElement("span");bar.style.flex=String(run.count);bar.style.background=run.truth===1?"#16a34a":"#64748b";el.appendChild(bar)}}
function receive(d){count++;ifCount+=Number(d.isolation_forest_prediction===1);lstmCount+=Number(d.lstm_prediction===1);hybridCount+=Number(d.combined_prediction===1);if(d.window_ready===true)windows++;$('processed').textContent=count;$('ifalerts').textContent=ifCount;$('lstmalerts').textContent=lstmCount;$('combined').textContent=hybridCount;$('windows').textContent=windows;$('window-status').textContent=d.window_ready?'Preparada':'Preparando';if(Number.isFinite(d.processing_latency_ns)){latencySum+=d.processing_latency_ns/1e6;latencyN++;$('avg-lat').textContent=fmt(latencySum/latencyN,2)+' ms'}
chart.data.labels.push(String(d.point_index ?? d.sequence_id ?? count));for(const [i,v] of [d.isolation_forest_score,d.isolation_forest_threshold,d.lstm_score,d.lstm_threshold].entries())chart.data.datasets[i].data.push(v==null?null:Number(v));if(chart.data.labels.length>80){chart.data.labels.shift();chart.data.datasets.forEach(s=>s.data.shift())}if(count%5===0 || d.combined_prediction===1)chart.update('none');
if(d.isolation_forest_prediction===1)addAlert(d,'Isolation Forest',d.isolation_forest_score);if(d.lstm_prediction===1)addAlert(d,'LSTM AE',d.lstm_score);if(d.combined_prediction===1)addAlert(d,'Híbrido',null);const uniqueAlert=d.isolation_forest_prediction===1||d.lstm_prediction===1||d.combined_prediction===1;if(uniqueAlert)uniqueAlertCount++;$('critical-count').textContent=uniqueAlertCount;updateValidation(d);updateComparison(d);
}

function receiveNAB(d){
    count++;
    ifCount += Number(d.isolation_forest_prediction === 1);
    lstmCount += Number(d.robust_prediction === 1);
    hybridCount += Number(d.combined_prediction === 1);

    $('processed').textContent = count;
    $('ifalerts').textContent = ifCount;
    $('lstmalerts').textContent = lstmCount;
    $('combined').textContent = hybridCount;

    $('windows').textContent = 'N/D';
    $('window-status').textContent = 'No aplica';
    $('avg-lat').textContent = 'N/D';

    chart.data.labels.push(String(d.point_index ?? count));
    chart.data.datasets[0].data.push(
        Number.isFinite(Number(d.value)) ? Number(d.value) : null
    );

    for(let i = 1; i < chart.data.datasets.length; i++){
        chart.data.datasets[i].data.push(null);
    }

    if(chart.data.labels.length > 80){
        chart.data.labels.shift();
        chart.data.datasets.forEach(s => s.data.shift());
    }

    if(count % 5 === 0 || d.combined_prediction === 1){
        chart.update('none');
    }

    if(d.isolation_forest_prediction === 1){
        addAlert(d, 'Isolation Forest', d.isolation_forest_score);
    }
    if(d.robust_prediction === 1){
        addAlert(d, 'Robust (mediana/MAD)', d.robust_score);
    }
    if(d.combined_prediction === 1){
        addAlert(d, 'Hibrido ' + validationPolicy.toUpperCase(), null);
    }

    if(
        d.isolation_forest_prediction === 1 ||
        d.robust_prediction === 1
    ){
        uniqueAlertCount++;
    }
    $('critical-count').textContent = uniqueAlertCount;

    const adapted = {
        ...d,
        lstm_prediction: d.robust_prediction
    };

    updateValidation(adapted);
    updateComparison(adapted);
}


function receiveHDFS(d) {
    count++;

    ifCount += Number(d.isolation_forest_prediction === 1);
    lstmCount += Number(d.lstm_prediction === 1);
    hybridCount += Number(d.combined_prediction === 1);

    $('processed').textContent = count;
    $('ifalerts').textContent = ifCount;
    $('lstmalerts').textContent = lstmCount;
    $('combined').textContent = hybridCount;

    $('windows').textContent = 'N/D';
    $('window-status').textContent = 'No aplica';
    $('avg-lat').textContent = 'N/D';

    chart.data.labels.push(String(d.point_index ?? count));

    const scores = [
        null,
        null,
        Number.isFinite(d.lstm_score) ? d.lstm_score : null,
        null
    ];

    scores.forEach((value, index) => {
        chart.data.datasets[index].data.push(value);
    });

    if (chart.data.labels.length > 80) {
        chart.data.labels.shift();
        chart.data.datasets.forEach(series => series.data.shift());
    }

    if (count % 5 === 0 || d.combined_prediction === 1) {
        chart.update('none');
    }

    if (d.isolation_forest_prediction === 1) {
        addAlert(d, 'Isolation Forest', null);
    }

    if (d.lstm_prediction === 1) {
        addAlert(d, 'LSTM Next-Event', d.lstm_score);
    }

    if (d.combined_prediction === 1) {
        addAlert(d, 'Politica ' + validationPolicy.toUpperCase(), null);
    }

    if (d.isolation_forest_prediction === 1 ||
        d.lstm_prediction === 1) {
        uniqueAlertCount++;
    }

    $('critical-count').textContent = uniqueAlertCount;

    updateValidation(d);
    updateComparison(d);
}


function receiveBGL(d) {
    count++;

    const ifPred = d.isolation_forest_prediction;
    const sgdPred = d.sgd_prediction;
    const selectedPred = d.combined_prediction;

    ifCount += Number(ifPred === 1);
    lstmCount += Number(sgdPred === 1);
    hybridCount += Number(selectedPred === 1);

    $('processed').textContent = count;
    $('ifalerts').textContent = ifCount;
    $('lstmalerts').textContent = lstmCount;
    $('combined').textContent = hybridCount;

    $('windows').textContent = 'N/D';
    $('window-status').textContent = 'No aplica';
    $('avg-lat').textContent = 'N/D';

    chart.data.labels.push(String(d.line_number ?? d.point_index ?? count));

    const values = [ifPred, null, sgdPred, selectedPred];

    values.forEach((value, index) => {
        chart.data.datasets[index].data.push(
            value === 0 || value === 1 ? value : null
        );
    });

    if (chart.data.labels.length > 80) {
        chart.data.labels.shift();
        chart.data.datasets.forEach(series => series.data.shift());
    }

    if (count % 5 === 0 || selectedPred === 1) {
        chart.update('none');
    }

    if (ifPred === 1) addAlert(d, 'Isolation Forest', null);
    if (sgdPred === 1) addAlert(d, 'SGD supervisado', null);
    if (selectedPred === 1) {
        addAlert(d, 'Politica ' + validationPolicy.toUpperCase(), null);
    }

    if (ifPred === 1 || sgdPred === 1) uniqueAlertCount++;
    $('critical-count').textContent = uniqueAlertCount;

    const adapted = {
        ...d,
        lstm_prediction: sgdPred
    };

    updateValidation(adapted);
    updateComparison(adapted);
}

function stop(){runId++;if(socket){const s=socket;socket=null;s.close()}$('start').disabled=['hdfs','bgl'].includes($('dataset').value)?availableSources[$('dataset').value]?.status!=='disponible':!$('machine').value;$('stop').disabled=true;$('stream-status').textContent='Detenida';setStreaming('stopped');}
$('stop').onclick=()=>{stop();setStatus('Reproducción detenida por el usuario.');};
$('reset').onclick=()=>{stop();clearView();setStatus('Vista reiniciada.');};
$('start').onclick=()=>{if(!['hdfs','bgl'].includes($('dataset').value)&&!$('machine').value)return;stop();clearView();const id=++runId;const limit=Number($('limit').value);if(!Number.isInteger(limit)||limit<30||limit>10000){setStatus('Ingresá entre 30 y 10000 observaciones.',true);return}const start_index=Number($('start-index').value);if(!Number.isSafeInteger(start_index)||start_index<0){setStatus('Ingresá un índice inicial válido.',true);return}validationPolicy=$('policy').value;validationPartition=$('partition').value;$('comparison-scope').textContent=validationPartition==='X_test'?'X_test · '+validationPolicy.toUpperCase()+' · desde '+start_index:'X_val · sin etiquetas';$('comparison-note').textContent=validationPolicy==='dual'?'DUAL: predicciones individuales de IF y LSTM. No existe predicción híbrida combinada; las primeras ventanas pueden no estar disponibles.':'Comparación sobre y_test. IF evalúa observaciones individuales; LSTM e híbrido requieren ventana temporal. N/D: denominador cero.';$('validation-scope').textContent=validationPartition==='X_test'?'X_test · '+validationPolicy.toUpperCase()+' · desde '+start_index:'X_val · sin etiquetas verificadas';$('validation-note').textContent=validationPartition==='X_test'?(validationPolicy==='dual'?'DUAL: esta sección muestra LSTM; la tabla inferior compara ambos modelos (sin combinación híbrida).':'Predicciones híbridas '+validationPolicy.toUpperCase()+' frente a etiquetas reales y_test. Precisión sin predicciones positivas: 0 por convención.'):'No se calculan métricas de clasificación en X_val porque no hay etiquetas y_test correspondientes.';if($('dataset').value === 'hdfs'){
    const scope = 'HDFS / historical_test / ' +
        validationPolicy.toUpperCase() +
        ' / desde ' + start_index;

    $('validation-scope').textContent = scope;
    $('comparison-scope').textContent = scope;

    $('validation-note').textContent =
        'Evaluacion de la politica seleccionada frente a etiquetas reales ' +
        'de sesiones de bloque HDFS. Metricas calculadas exclusivamente ' +
        'sobre las observaciones reproducidas.';

    $('comparison-note').textContent =
        'Comparacion historica de Isolation Forest, LSTM Next-Event y ' +
        'la politica seleccionada sobre las mismas sesiones etiquetadas. ' +
        'No se ejecuta inferencia online.';
}
const isNABReplay = $('dataset').value === 'nab';
if(isNABReplay){
    const scope = 'NAB AWS ? historical_test ? ' +
        validationPolicy.toUpperCase() + ' ? desde ' + start_index;

    $('comparison-scope').textContent = scope;
    $('validation-scope').textContent = scope;

    $('comparison-note').textContent =
        validationPolicy === 'dual'
        ? 'DUAL: se comparan Isolation Forest y Robust por separado. No existe prediccion hibrida combinada.'
        : 'Evaluacion historica frente a label_window. Se comparan Isolation Forest, Robust y la politica ' +
          validationPolicy.toUpperCase() +
          '. Estas metricas no son la puntuacion oficial del benchmark NAB.';

    $('validation-note').textContent =
        validationPolicy === 'dual'
        ? 'DUAL: esta seccion muestra Robust; la tabla compara ambos detectores por separado.'
        : 'Matriz de confusion de la politica ' +
          validationPolicy.toUpperCase() +
          ' frente a label_window. Evaluacion historica, no inferencia en linea.';
}
if ($('dataset').value === 'bgl') {
    const scope = 'BGL / historical_test / ' +
        validationPolicy.toUpperCase() +
        ' / desde ' + start_index;

    $('validation-scope').textContent = scope;
    $('comparison-scope').textContent = scope;

    $('validation-note').textContent =
        'Matriz de confusion de la politica ' +
        validationPolicy.toUpperCase() +
        ' frente a etiquetas reales de eventos BGL. ' +
        'Metricas calculadas solo sobre los eventos reproducidos. ' +
        'No se ejecuta inferencia online.';

    $('comparison-note').textContent =
        'Comparacion de Isolation Forest, SGD supervisado y la politica ' +
        validationPolicy.toUpperCase() +
        ' sobre los mismos eventos etiquetados del conjunto historical_test. ' +
        'Las predicciones fueron calculadas previamente. ' +
        'No hay scores disponibles en esta reproduccion.';
}
const dataset=$('dataset').value;const selectedMachine=$('machine').value;const payload={start_index,partition:$('partition').value,policy:$('policy').value,limit};if(dataset==='nasa'){payload.channel=selectedMachine;}else if(dataset==='nab'){payload.series=selectedMachine;}else if(!['hdfs','bgl'].includes(dataset)){payload.machine=selectedMachine;}$('active-partition').textContent=payload.partition;$('selected-source').textContent=['hdfs','bgl'].includes(dataset)?dataset.toUpperCase():dataset.toUpperCase()+' / '+selectedMachine;$('start').disabled=true;$('stop').disabled=false;$('stream-status').textContent='Conectando';setStreaming('connecting');setStatus(dataset==='bgl'?'Preparando reproduccion de predicciones historicas BGL...':dataset==='hdfs'?'Preparando reproduccion de predicciones historicas HDFS...':dataset==='nab'?'Preparando reproduccion de predicciones historicas NAB AWS...':'Calculando inferencia y preparando contexto temporal '+dataset.toUpperCase()+'...');const endpoint=dataset==='bgl'?'/ws/replay/bgl':dataset==='hdfs'?'/ws/replay/hdfs':dataset==='nasa'?'/ws/replay/nasa':dataset==='nab'?'/ws/replay/nab':'/ws/replay/smd';const ws=new WebSocket((location.protocol==='https:'?'wss://':'ws://')+location.host+endpoint);socket=ws;ws.onopen=()=>{if(id!==runId)return;ws.send(JSON.stringify(payload));$('stream-status').textContent='Activa';setStreaming('active');setStatus(dataset==='bgl'?'Reproduciendo predicciones historicas BGL...':dataset==='nab'?'Reproduciendo predicciones historicas NAB AWS...':dataset==='hdfs'?'Reproduciendo predicciones historicas HDFS...':'Reproduciendo resultados de inferencia...')};ws.onmessage=e=>{if(id!==runId)return;let msg;try{msg=JSON.parse(e.data)}catch(_){setStatus('Respuesta WebSocket no válida.',true);stop();return}if(msg.type==='observation'&&msg.data){if(dataset==='bgl'){receiveBGL(msg.data)}else if(dataset==='hdfs'){receiveHDFS(msg.data)}else if(dataset==='nab'){receiveNAB(msg.data)}else{receive(msg.data)}window.dispatchEvent(new CustomEvent('apetria:observation',{detail:{dataset,observation:msg.data,policy:validationPolicy,partition:validationPartition}}));}else if(msg.type==='completed'){chart.update('none');drawTruthTimeline();renderComparison();setStatus('Reproducción completada: '+count+' observaciones.');$('stream-status').textContent='Completada';setStreaming('done');$('stop').disabled=true;$('start').disabled=false;}else if(msg.type==='error'){setStatus('Error del backend: '+(msg.message||'sin detalle'),true);stop();setStreaming('error')}};ws.onerror=()=>{if(id===runId){setStatus('Error de conexión WebSocket.',true);setStreaming('error')}};ws.onclose=(event)=>{if(event.code===4401){window.location.reload();return;}if(id===runId){socket=null;$('start').disabled=false;$('stop').disabled=true;if($('stream-status').textContent==='Activa'){$('stream-status').textContent='Cerrada';setStreaming('stopped')}}};};

let availableSources = {};


function updateDatasetLabels() {
    const name = {
        smd: 'SMD',
        nasa: 'NASA SMAP/MSL',
        nab: 'NAB AWS',
        hdfs: 'HDFS',
        bgl: 'BGL'
    }[$('dataset').value] || 'SMD';

    for (const id of [
        'validation-dataset',
        'comparison-dataset',
        'model-dataset',
        'partition-dataset'
    ]) {
        $(id).textContent = name;
    }
}

// Nombres precisos: una senal individual no implica una alerta final.
function updatePolicyKpi() {
    const dataset = $('dataset').value;
    const policy = $('policy').value;
    const card = $('combined')?.closest('.kpi');
    if (!card) return;
    const title = card.querySelector('.kpi-title');
    const caption = card.querySelector('.kpi-caption');
    const foot = card.querySelector('.kpi-foot');
    const dual = policy === 'dual' && !['hdfs', 'bgl'].includes(dataset);
    if (title) title.textContent = dual ? 'Sin combinación (DUAL)' : 'Alertas de política';
    if (caption) caption.textContent = dual ? 'Detectores individuales' : 'Política ' + policy.toUpperCase();
    if (foot) foot.textContent = dual
        ? 'Sin decisión final combinada'
        : dataset === 'bgl'
            ? 'Predicciones históricas por evento de log'
            : dataset === 'hdfs'
                ? 'Predicciones históricas por sesión'
                : dataset === 'nab'
                    ? 'Predicciones históricas de la política'
                    : 'Predicciones combinadas';
}

function updateDatasetSelection() {
    const dataset = $('dataset').value;

    const hdfsMode = dataset === 'hdfs';

    const combinedCard = $('combined')?.closest('.kpi');


    if (hdfsMode) {
        $('replay-explanation').textContent =
            'Modo: reproduccion historica de predicciones HDFS. ' +
            'Inferencia online: NO. Cada observacion corresponde a una ' +
            'sesion de bloque de historical_test. Isolation Forest y ' +
            'LSTM Next-Event fueron evaluados previamente; el WebSocket ' +
            'transmite sus resultados almacenados. La velocidad de ' +
            'reproduccion no representa throughput de inferencia.';

        const modelDataset = $('model-dataset');
        const modelFoot = modelDataset?.closest('.kpi-foot');

        if (modelFoot) {
            modelFoot.firstChild.textContent =
                'Predicciones historicas - ';
        }
    } else {
        $('replay-explanation').textContent =
            'El indice inicial conserva el contexto temporal necesario ' +
            'para las ventanas LSTM; el procesamiento se limita al ' +
            'segmento solicitado y su preparacion. Inferencia calculada ' +
            'sobre datos reales antes de la transmision. El WebSocket ' +
            'reproduce las predicciones; su tasa de recepcion no mide ' +
            'throughput de inferencia en linea.';

        const modelDataset = $('model-dataset');
        const modelFoot = modelDataset?.closest('.kpi-foot');

        if (modelFoot) {
            modelFoot.firstChild.textContent = ($('dataset').value === 'bgl' ? 'Predicciones historicas - ' : 'Modelo entrenado - ');
        }
    }

    updateDatasetLabels();

    const partition = $('partition');
    const requiredPartitions = dataset === 'nab' || dataset === 'hdfs' || dataset === 'bgl'
        ? [['historical_test', 'Evaluacion historica']]
        : [['X_test', 'X_test'], ['X_val', 'X_val']];

    partition.replaceChildren();

    for (const [value, label] of requiredPartitions) {
        const option = document.createElement('option');
        option.value = value;
        option.textContent = label;
        partition.appendChild(option);
    }

    const isNAB = dataset === 'nab';
    const isHDFS = dataset === 'hdfs';
    const isBGL = dataset === 'bgl';

    const policySelect = $('policy');
    const previousPolicy = policySelect.value;

    const policyOptions = isBGL
        ? [
            ['sgd', 'SGD supervisado'],
            ['if', 'Isolation Forest'],
            ['and', 'AND'],
            ['or', 'OR']
          ]
        : isHDFS
        ? [
            ['lstm', 'LSTM Next-Event'],
            ['if', 'Isolation Forest'],
            ['and', 'AND'],
            ['or', 'OR']
          ]
        : [
            ['and', 'AND'],
            ['or', 'OR'],
            ['dual', 'DUAL']
          ];

    policySelect.replaceChildren();

    for (const [value, label] of policyOptions) {
        const option = document.createElement('option');
        option.value = value;
        option.textContent = label;
        policySelect.appendChild(option);
    }

    policySelect.value = isBGL
        ? 'sgd'
        : isHDFS
        ? 'lstm'
        : (policyOptions.some(([value]) => value === previousPolicy)
            ? previousPolicy
            : 'and');

    updatePolicyKpi();

    const modelBadge = document.querySelector('#active-models span');
    const secondCard = $('second-model-card');
    const chartTitle = document.querySelector(
        '#metrics-chart-card .panel-title'
    );

    if (modelBadge) {
        modelBadge.textContent = isBGL
            ? 'IF + SGD supervisado'
            : isHDFS
            ? 'IF + LSTM Next-Event'
            : isNAB
                ? 'IF + Robust (mediana/MAD)'
                : 'IF + LSTM AE';
    }

    if (secondCard) {
        secondCard.querySelector('.kpi-title').textContent =
            isBGL ? 'Señales SGD' : (isHDFS ? 'Señales LSTM Next-Event' : (isNAB ? 'Señales Robust' : 'Señales LSTM AE'));

        secondCard.querySelector('.kpi-caption').textContent =
            isBGL ? 'SGD supervisado' : (isHDFS ? 'LSTM Next-Event' : (isNAB ? 'Mediana / MAD' : 'LSTM Autoencoder'));

        secondCard.querySelector('.kpi-foot').textContent =
            (isNAB || isHDFS || isBGL) ? 'Predicciones historicas' : 'Tras ventana temporal';
    }

    if (chartTitle) {
        chartTitle.textContent = isNAB
            ? 'Telemetria AWS - Valores historicos'
            : 'Scores y umbrales de deteccion';
    }

    delete chart.options.scales.yIF.min;
    delete chart.options.scales.yIF.max;

    const chartLabels = [
        'Isolation Forest',
        'Umbral IF',
        'LSTM Autoencoder',
        'Umbral LSTM'
    ];

    chart.data.datasets.forEach((series, index) => {
        series.label = isNAB
            ? (index === 0 ? 'Valor de telemetria AWS' : chartLabels[index])
            : chartLabels[index];

        series.hidden = isNAB && index !== 0;
        series.yAxisID = isNAB ? 'yIF' : (index < 2 ? 'yIF' : 'yLSTM');
    });

    chart.options.plugins.legend.labels.filter = isNAB
        ? (item) => item.datasetIndex === 0
        : () => true;

    chart.options.scales.yLSTM.display = !isNAB;
    chart.options.scales.yIF.title = {
        display: isNAB,
        text: 'Valor observado'
    };

    if (isBGL) {
        const bglLabels = [
            'Isolation Forest - prediccion binaria',
            'No disponible',
            'SGD supervisado - prediccion binaria',
            'Politica seleccionada - prediccion binaria'
        ];

        chart.data.datasets.forEach((series, index) => {
            series.label = bglLabels[index];
            series.hidden = index === 1;
            series.yAxisID = 'yIF';
        });

        chart.options.plugins.legend.labels.filter =
            item => item.datasetIndex !== 1;

        chart.options.scales.yIF.display = true;
        chart.options.scales.yIF.min = 0;
        chart.options.scales.yIF.max = 1;
        chart.options.scales.yIF.title = {
            display: true,
            text: 'Prediccion binaria (0 normal, 1 anomalia)'
        };

        chart.options.scales.yLSTM.display = false;

        if (chartTitle) {
            chartTitle.textContent =
                'BGL - Predicciones binarias historicas por evento';
        }
    } else if (isHDFS) {
        chart.data.datasets.forEach((series, index) => {
            series.label = index === 2
                ? 'LSTM Next-Event - NLL media por sesion'
                : chartLabels[index];
            series.hidden = index !== 2;
            series.yAxisID = 'yLSTM';
        });

        chart.options.plugins.legend.labels.filter =
            item => item.datasetIndex === 2;

        chart.options.scales.yIF.display = false;
        chart.options.scales.yLSTM.display = true;

        if (chartTitle) {
            chartTitle.textContent =
                'HDFS - Score LSTM Next-Event por sesion';
        }
    } else {
        chart.options.scales.yIF.display = true;
    }

    chart.update('none');

    const windowCounter = $('windows');
    const windowFoot = windowCounter?.closest('.kpi-foot');

    if (windowFoot) {
        windowFoot.childNodes.forEach(node => {
            if (node.nodeType === Node.TEXT_NODE) {
                node.textContent = (isNAB || isHDFS || isBGL)
                    ? 'Ventanas temporales: '
                    : 'Ventanas LSTM: ';
            }
        });
    }

    if (isNAB || isHDFS || isBGL) {
        $('windows').textContent = 'N/D';
        $('window-status').textContent = 'No aplica';
        $('avg-lat').textContent = 'N/D';
    }

    const explanation = $('replay-explanation');

    if (explanation) {
        if (!explanation.dataset.originalText) {
            explanation.dataset.originalText = explanation.textContent;
        }

        if (isBGL) {
            explanation.textContent =
                'BGL: reproduccion historica de predicciones de ' +
                'Isolation Forest y SGD supervisado sobre eventos ' +
                'de historical_test. No se ejecuta inferencia online. ' +
                'No hay scores ni latencias de inferencia disponibles. ' +
                'La velocidad del WebSocket no representa throughput ' +
                'de inferencia.';
        } else if (isHDFS) {
            explanation.textContent =
                'Modo: reproduccion historica de predicciones HDFS. ' +
                'Inferencia online: NO. Cada observacion corresponde a una ' +
                'sesion de bloque de historical_test. Isolation Forest y ' +
                'LSTM Next-Event fueron evaluados previamente; el WebSocket ' +
                'transmite sus resultados almacenados. La velocidad de ' +
                'reproduccion no representa throughput de inferencia.';
        } else if (isNAB) {
            explanation.textContent =
                'NAB AWS: reproduccion de predicciones historicas de ' +
                'Isolation Forest y Robust (mediana/MAD). La evaluacion ' +
                'utiliza label_window; no corresponde a la puntuacion ' +
                'oficial NAB. La velocidad de recepcion WebSocket no mide ' +
                'throughput de inferencia en linea.';
        } else {
            explanation.textContent = explanation.dataset.originalText;
        }
    }

    const source = availableSources[dataset];
    const machines = source?.machines || [];

    $('machine').replaceChildren();
    $('machine').disabled = dataset === 'hdfs' || dataset === 'bgl';

    for (const name of machines) {
        const option = document.createElement('option');
        option.value = name;
        option.textContent = name;
        $('machine').appendChild(option);
    }

    $('machine-label').textContent = {
        smd: 'Maquina SMD',
        nasa: 'Canal NASA',
        nab: 'Serie AWS',
        hdfs: 'Unidad: sesi?n de bloque',
        bgl: 'Unidad: evento de log'
    }[dataset] || 'Fuente';

    $('selected-source').textContent = dataset.toUpperCase();

    const hdfsAvailable =
        dataset === 'hdfs' && source?.status === 'disponible';
    const bglAvailable =
        dataset === 'bgl' && source?.status === 'disponible';

    $('start').disabled = !(hdfsAvailable || bglAvailable) && machines.length === 0;

    if ((isHDFS || isBGL) && combinedCard) {
        const caption = combinedCard.querySelector('.kpi-caption');
        if (caption) {
            caption.textContent =
                'Politica ' + $('policy').value.toUpperCase();
        }
    }

    setStatus(
        bglAvailable
            ? 'BGL disponible: ' + source.total_events +
              ' eventos de log. Reproduccion historica.'
            : hdfsAvailable
            ? 'HDFS disponible: ' + source.total_sessions +
              ' sesiones de bloques. Reproduccion historica.'
            : dataset.toUpperCase() + ' disponible: ' +
              machines.length + ' canales/maquinas. ' +
              'Selecciona los parametros para iniciar.'
    );
}

$('dataset').addEventListener('change', () => {
    stop();
    clearView();
    updateDatasetSelection();
});

$('policy').addEventListener('change', () => {
    stop();
    clearView();
    updatePolicyKpi();

    if (['hdfs', 'bgl'].includes($('dataset').value)) {
        const combinedCard = $('combined')?.closest('.kpi');
        const caption = combinedCard?.querySelector('.kpi-caption');

        if (caption) {
            caption.textContent =
                'Politica ' + $('policy').value.toUpperCase();
        }

        $('validation-scope').textContent =
            $('dataset').value.toUpperCase() +
            ' / historical_test / esperando reproduccion';

        $('comparison-scope').textContent =
            $('dataset').value.toUpperCase() +
            ' / historical_test / esperando reproduccion';
    }

    setStatus('Politica actualizada. Inicia una nueva reproduccion.');
});

async function init() {
    try {
        const response = await fetch(API + '/api/sources');

        if (!response.ok) {
            throw new Error('HTTP ' + response.status);
        }

        const payload = await response.json();
        const sources = payload.sources || [];

        availableSources = Object.fromEntries(
            sources.map(source => [source.id, source])
        );

        $('sources').replaceChildren();

        for (const source of sources) {
            const pill = document.createElement('span');

            pill.className =
                'dataset-pill' +
                (source.status === 'disponible' ? ' available' : '');

            pill.textContent =
                source.id.toUpperCase() + ' - ' +
                source.status.replaceAll('_', ' ');

            $('sources').appendChild(pill);
        }

        for (const option of $('dataset').options) {
            const source = availableSources[option.value];

            option.disabled =
                !source || source.status !== 'disponible';
        }

        if ($('dataset').selectedOptions[0]?.disabled) {
            const first = Array.from($('dataset').options)
                .find(option => !option.disabled);

            if (first) {
                $('dataset').value = first.value;
            }
        }

        // Preferencias persistidas por el backend; la fuente sin artefactos no se fuerza.
        let preferences = null;
        try {
            const settingsResponse = await fetch(API + '/api/config');
            if (settingsResponse.ok) preferences = await settingsResponse.json();
        } catch (_) { /* Mantener los controles originales si no hay preferencias */ }
        if (preferences) {
            const preferred = Array.from($('dataset').options).find(o => o.value === preferences.default_dataset && !o.disabled);
            if (preferred) $('dataset').value = preferred.value;
            if (Number.isInteger(preferences.default_limit)) $('limit').value=String(preferences.default_limit);
            if (preferences.theme === 'dark' || preferences.theme === 'light') theme(preferences.theme);
        }
        updateDatasetSelection();
        if (preferences) {
            const policy = Array.from($('policy').options).find(o => o.value === preferences.default_policy);
            if (policy) $('policy').value = policy.value;
        }

        $('connection').textContent = 'API CONECTADA';
        $('api-badge').classList.remove('error');

    } catch (error) {
        $('connection').textContent = 'API DESCONECTADA';
        $('api-badge').classList.add('error');
        $('start').disabled = true;

        setStatus(
            'No se pudo conectar con FastAPI: ' + error.message,
            true
        );
    }
}

if(window.lucide)lucide.createIcons();init();
})();
