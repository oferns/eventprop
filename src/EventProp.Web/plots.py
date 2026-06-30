"""Plotly chart builders for EventProp Visualizer."""
import numpy as np
import plotly.graph_objects as go

CY_GREEN, PY_RED, BOTH_BLUE, ORANGE = "#22c55e", "#ef4444", "#3b82f6", "#f59e0b"


def learning_curve(values, title, y_label, log_y=True):
    fig = go.Figure()
    for v in values:
        fig.add_trace(go.Scatter(x=np.arange(1,len(v)+1),y=v,mode="lines",line=dict(color="#888",width=1),opacity=0.15,showlegend=False))
    mean, std = np.mean(values,axis=0), np.std(values,axis=0)
    fig.add_trace(go.Scatter(x=np.arange(1,len(mean)+1),y=mean,mode="lines",line=dict(color="#2563eb",width=3),name="Mean"))
    fig.add_trace(go.Scatter(x=np.arange(1,len(mean)+1),y=mean+std,mode="lines",fill="tonexty",fillcolor="rgba(37,99,235,0.15)",line=dict(width=0),name="+-1s"))
    fig.update_layout(title=title,xaxis_title="Epoch",yaxis_title=y_label,height=350,margin=dict(l=65,r=20,t=45,b=50),yaxis=dict(type="log" if log_y else "linear"))
    return fig


def raster(spikes, title, tmax):
    ts = np.array(spikes.times)*1e3; sr = np.array(spikes.sources)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts,y=sr,mode="markers",marker=dict(size=4,color="#2563eb"),hovertemplate="Neuron %{y} @ %{x:.2f}ms<extra></extra>"))
    n_max = int(sr.max())+1 if len(sr)>0 else 1
    fig.update_layout(title=title,xaxis_title="Time (ms)",yaxis_title="Neuron",height=min(350,max(120,n_max*2)),margin=dict(l=55,r=15,t=35,b=35),xaxis=dict(range=[0,tmax]),yaxis=dict(autorange="reversed"))
    return fig


def heatmap(w, title):
    m = np.abs(w).max() if np.abs(w).max()>0 else 1.0
    fig = go.Figure(data=go.Heatmap(z=w,colorscale="RdBu_r",zmin=-m,zmax=m,hovertemplate="(%{y},%{x}): %{z:.4f}<extra></extra>"))
    fig.update_layout(title=title,xaxis_title="Post-synaptic",yaxis_title="Pre-synaptic",height=min(500,w.shape[0]+100),margin=dict(l=80,r=40,t=45,b=60))
    return fig


def raster_compare(cy_s, py_s, title, tmax):
    tol_ms = 0.5
    cy_nrn, py_nrn = {}, {}
    for s,t in zip(cy_s.sources,cy_s.times): cy_nrn.setdefault(int(s),[]).append(float(t)*1e3)
    for s,t in zip(py_s.sources,py_s.times): py_nrn.setdefault(int(s),[]).append(float(t)*1e3)
    gx,gy,rx,ry,bx,by = [],[],[],[],[],[]
    for nrn in sorted(set(cy_nrn)|set(py_nrn)):
        ct,pt = cy_nrn.get(nrn,[]),py_nrn.get(nrn,[])
        used = set()
        for tc in ct:
            matched = False
            for j,tp in enumerate(pt):
                if abs(tc-tp)<=tol_ms and j not in used: bx.append(tc);by.append(nrn);used.add(j);matched=True;break
            if not matched: gx.append(tc);gy.append(nrn)
        for j,tp in enumerate(pt):
            if j not in used: rx.append(tp);ry.append(nrn)
    fig = go.Figure()
    if bx: fig.add_trace(go.Scatter(x=bx,y=by,mode="markers",marker=dict(size=6,color=BOTH_BLUE),name="Both agree"))
    if gx: fig.add_trace(go.Scatter(x=gx,y=gy,mode="markers",marker=dict(size=6,color=CY_GREEN),name="Cython only"))
    if rx: fig.add_trace(go.Scatter(x=rx,y=ry,mode="markers",marker=dict(size=6,color=PY_RED),name="Pybind11 only"))
    all_nrn = sorted(set(cy_nrn)|set(py_nrn))
    n_max = max(all_nrn)+1 if all_nrn else 1
    fig.update_layout(title=title,xaxis_title="Time (ms)",yaxis_title="Neuron",height=min(400,max(200,n_max*2)),margin=dict(l=55,r=15,t=35,b=35),xaxis=dict(range=[0,tmax]),yaxis=dict(autorange="reversed"),legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="center",x=0.5))
    return fig


def fst_compare(cy_l, py_l, title):
    tol_ms = 0.5; n = cy_l.parameters.n
    cy_fst = np.full(n,np.nan); py_fst = np.full(n,np.nan)
    for t,s in zip(cy_l.post_batch[0].times,cy_l.post_batch[0].sources):
        i=int(s)
        if np.isnan(cy_fst[i]) or float(t)<cy_fst[i]: cy_fst[i]=float(t)
    for t,s in zip(py_l.post_batch[0].times,py_l.post_batch[0].sources):
        i=int(s)
        if np.isnan(py_fst[i]) or float(t)<py_fst[i]: py_fst[i]=float(t)
    cy_only,py_only,both = [],[],[]
    for nrn in range(n):
        c_fire,p_fire = not np.isnan(cy_fst[nrn]),not np.isnan(py_fst[nrn])
        if c_fire and p_fire:
            if abs((cy_fst[nrn]-py_fst[nrn])*1e3)<=tol_ms: both.append(nrn)
            else: cy_only.append(nrn); py_only.append(nrn)
        elif c_fire: cy_only.append(nrn)
        elif p_fire: py_only.append(nrn)
    fig = go.Figure()
    if both: fig.add_trace(go.Bar(x=both,y=cy_fst[both]*1e3,marker_color=BOTH_BLUE,name="Both agree"))
    if cy_only: fig.add_trace(go.Bar(x=cy_only,y=cy_fst[cy_only]*1e3,marker_color=CY_GREEN,name="Cython only"))
    if py_only: fig.add_trace(go.Bar(x=py_only,y=py_fst[py_only]*1e3,marker_color=PY_RED,name="Pybind11 only"))
    fig.update_layout(title=title,xaxis_title="Neuron",yaxis_title="First Spike Time (ms)",height=300,margin=dict(l=55,r=15,t=35,b=35),barmode="group",legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="center",x=0.5))
    return fig


def grad_scatter(cy_grad, py_grad, title):
    diff = np.abs(cy_grad - py_grad)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cy_grad,y=py_grad,mode="markers",marker=dict(size=8,color=diff,colorscale="YlOrRd",showscale=True,colorbar=dict(title="|Diff|")),hovertemplate="Neuron %{text}<br>Cy: %{x:.4f}<br>Py: %{y:.4f}<extra></extra>",text=[f"#{i}" for i in range(len(cy_grad))]))
    m = max(cy_grad.max(),py_grad.max()) if (cy_grad.max()>0 or py_grad.max()>0) else 1.0
    fig.add_shape(type="line",x0=0,y0=0,x1=m*1.05,y1=m*1.05,line=dict(color="#888",width=1,dash="dash"))
    corr = np.corrcoef(cy_grad,py_grad)[0,1] if len(cy_grad)>1 else 0
    fig.update_layout(title=f"{title} (corr={corr:.3f})",xaxis_title="Cython |grad| max",yaxis_title="Pybind11 |grad| max",height=400,margin=dict(l=65,r=80,t=55,b=55))
    return fig


def grad_divergence(cy_grad, py_grad, title):
    diff = np.abs(cy_grad - py_grad)
    top_idx = np.argsort(diff)[-30:][::-1]
    colors = [CY_GREEN if cy_grad[i]>=py_grad[i] else PY_RED for i in top_idx]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f"#{i}" for i in top_idx],y=diff[top_idx],marker_color=colors,hovertemplate="Neuron %{x}<br>|Cy-Py|: %{y:.4f}<extra></extra>"))
    fig.update_layout(title=title,xaxis_title="Neuron",yaxis_title="Gradient |Difference|",height=300,margin=dict(l=55,r=15,t=45,b=55),showlegend=False)
    return fig

