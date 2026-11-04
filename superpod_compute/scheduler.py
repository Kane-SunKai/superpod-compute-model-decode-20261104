"""Operator-independent event service and accelerated H/R/C or H/R/S/C pipelines."""
from dataclasses import asdict, dataclass
from .contracts import positive, integer

@dataclass(frozen=True)
class Event:
    name: str
    duration_s: float
    resources: tuple = ()
    dependencies: tuple = ()
    kind: str = 'compute'
    bytes: int = 0

class EventScheduler:
    """Deterministic, in-list topological scheduling, one server per resource ID.

    Dependencies can encode buffer release. This is not a dynamic GPU thread scheduler.
    A resource reservation spans the event duration. Multiple resources are reserved together.
    """
    def run(self, events):
        completed={};free={};result=[]
        for event in events:
            positive(event.duration_s,event.name,True)
            if event.name in completed:raise ValueError('Duplicate event: '+event.name)
            if any(dep not in completed for dep in event.dependencies):raise ValueError('Events must be topologically ordered')
            start=max([0.0]+[completed[d] for d in event.dependencies]+[free.get(r,0.0) for r in event.resources])
            end=start+event.duration_s
            completed[event.name]=end
            for resource in event.resources:free[resource]=end
            result.append({**asdict(event),'start_s':start,'end_s':end})
        return result

def pipeline(h, r, c, count, l2_buffers, shared_buffers, tail=None, *, accelerate=True, shared_port=False, trace_limit=0):
    for v in (h,r,c):positive(v,'stage duration',True)
    integer(count,'count',0)
    integer(trace_limit,'trace_limit',0)
    if l2_buffers not in (1,2) or shared_buffers not in (1,2):raise ValueError('invalid buffer count')
    if tail is not None:
        if len(tail)!=3:raise ValueError('tail must contain H/R/C durations')
        for v in tail:positive(v,'tail duration',True)
    segments=[(count,h,r,c)]
    if tail is not None:segments.append((1,*tail))
    return pipeline_segments(segments,l2_buffers,shared_buffers,accelerate=accelerate,shared_port=shared_port,trace_limit=trace_limit)


def pipeline_segments(segments,l2_buffers,shared_buffers,*,accelerate=True,shared_port=False,trace_limit=0):
    """Piecewise H/R/C service, retaining resource and buffer state at boundaries.

    Each tuple is (panel_count, H_seconds, R_seconds, C_seconds). In particular,
    a reused input has H=0 but still needs its R and C service. A zero-H panel
    may conservatively reserve an input window; it never bypasses R or C.
    """
    if l2_buffers not in (1,2) or shared_buffers not in (1,2):raise ValueError('invalid buffer count')
    integer(trace_limit,'trace_limit',0)
    H=R=C=0.0;lf=[0.0]*l2_buffers;sf=[0.0]*shared_buffers;i=0;skipped=0;trace=[]
    def step(hh,rr,cc):
        nonlocal H,R,C,i
        ls=i%l2_buffers;ss=i%shared_buffers
        h0=max(H,lf[ls],R if shared_port else 0.0);h1=h0+hh
        r0=max(h1,R,sf[ss]);r1=r0+rr
        c0=max(r1,C);c1=c0+cc
        H,R,C=h1,r1,c1;lf[ls]=r1;sf[ss]=c1
        if len(trace)<trace_limit:
            trace.append({'panel':i,'l2_slot':ls,'shared_slot':ss,'H_start_s':h0,'H_end_s':h1,'R_start_s':r0,'R_end_s':r1,'C_start_s':c0,'C_end_s':c1})
        i+=1
    for count,h,r,c in segments:
        integer(count,'segment count',0)
        for value in (h,r,c):positive(value,'segment duration',True)
        stop=i+count;previous=None
        while i<stop:
            step(h,r,c)
            if accelerate and i%2==0:
                state=[H,R,C,*lf,*sf]
                if previous is not None:
                    deltas=[a-b for a,b in zip(state,previous)];delta=deltas[0]
                    if delta>0 and all(abs(d-delta)<=max(1e-18,abs(delta)*1e-10) for d in deltas):
                        cycles=(stop-i)//2
                        if cycles and len(trace)>=trace_limit:
                            shift=cycles*delta;H+=shift;R+=shift;C+=shift;lf=[v+shift for v in lf];sf=[v+shift for v in sf];i+=2*cycles;skipped+=2*cycles
                previous=[H,R,C,*lf,*sf]
    return {'duration_s':C,'H_end_s':H,'R_end_s':R,'C_end_s':C,'panels':i,'skipped_panels':skipped,'trace':trace}


def pipeline_events(triplets, l2_buffers, shared_buffers, shared_port=False):
    """Uncompressed DAG, useful for independent verification and extension examples."""
    events=[]
    for i,(h,r,c) in enumerate(triplets):
        hdeps=(f'R{i-l2_buffers}',) if i>=l2_buffers else ()
        rdeps=(f'H{i}',)+((f'C{i-shared_buffers}',) if i>=shared_buffers else ())
        cdeps=(f'R{i}',)+((f'C{i-1}',) if i else ())
        events.extend([Event(f'H{i}',h,('transfer' if shared_port else 'HBM',),hdeps,'load'),Event(f'R{i}',r,('transfer' if shared_port else 'L2',),rdeps,'load'),Event(f'C{i}',c,('matrix',),cdeps,'compute')])
    return events


def output_pipeline(segments,shared_buffers,l2_buffers,*,shared_port=False,trace_limit=0,accelerate=True):
    """W: accumulators -> Shared, L: output -> L2, D: L2 -> HBM.

    FP16 conversion has zero separate cost by policy. Output Shared reads are
    deliberately not timed. Shared slots live W.start..L.end; L2 slots live
    L.start..D.end. Shared write and the output port are independent resources.
    shared_port serializes L and D, not W. Slots are per output wave/chunk round.
    """
    if shared_buffers not in (1,2) or l2_buffers not in (1,2):raise ValueError('invalid output buffer count')
    integer(trace_limit,'trace_limit',0)
    ends=[0.0]*3;sf=[0.0]*shared_buffers;lf=[0.0]*l2_buffers;i=skipped=0;trace=[]
    for count,w,l,d in segments:
        integer(count,'segment count',0)
        for value in (w,l,d):positive(value,'output stage duration',True)
        stop=i+count;previous=None
        while i<stop:
            ss=i%shared_buffers;ls=i%l2_buffers
            w0=max(ends[0],sf[ss]);w1=w0+w
            l0=max(w1,ends[1],lf[ls],ends[2] if shared_port else 0);l1=l0+l
            d0=max(l1,ends[2]);d1=d0+d
            ends=[w1,l1,d1];sf[ss]=l1;lf[ls]=d1
            if len(trace)<trace_limit:
                trace.append({'panel':i,'shared_slot':ss,'l2_slot':ls,
                    'W_start_s':w0,'W_end_s':w1,'L_start_s':l0,'L_end_s':l1,'D_start_s':d0,'D_end_s':d1})
            i+=1
            if accelerate and i%2==0:
                state=ends+sf+lf
                if previous is not None:
                    deltas=[a-b for a,b in zip(state,previous)];delta=deltas[0]
                    if delta>0 and all(abs(v-delta)<=max(1e-18,abs(delta)*1e-10) for v in deltas):
                        repeats=(stop-i)//2
                        if repeats and len(trace)>=trace_limit:
                            offset=repeats*delta;ends=[x+offset for x in ends];sf=[x+offset for x in sf];lf=[x+offset for x in lf]
                            i+=2*repeats;skipped+=2*repeats
                previous=ends+sf+lf
    return {'duration_s':ends[2],'W_end_s':ends[0],'L_end_s':ends[1],'D_end_s':ends[2],
            'panels':i,'skipped_panels':skipped,'trace':trace,'stages':['register-to-Shared','L2-write','L2-read/HBM-drain']}

def operand_pipeline(segments,l2_buffers,shared_buffers,register_buffers,*,shared_port=False,local_shared_port=False,trace_limit=0,accelerate=True):
    """H: HBM->L2, R: L2->Shared, S: Shared->registers, C: matrix.

    Input window frees at R end, Shared frees at S end, operand registers free
    at C end. The persistent C accumulator is a distinct register allocation.
    LDS-read and input-write service use independent ports by default. With
    local_shared_port=True, R (input store) and S (operand load) share a server.
    """
    buffers=(l2_buffers,shared_buffers,register_buffers)
    if any(x not in (1,2) for x in buffers):raise ValueError('invalid buffer count')
    ends=[0.0]*4;slots=[[0.0]*n for n in buffers];i=0;skipped=0;trace=[]
    for number,h,r,s,c in segments:
        integer(number,'segment count',0)
        durations=(h,r,s,c)
        for x in durations:positive(x,'stage duration',True)
        stop=i+number;previous=None
        while i<stop:
            start=[];finish=[]
            for stage,dt in enumerate(durations):
                available=max(ends[stage],finish[-1] if stage else 0)
                if stage<3:available=max(available,slots[stage][i%buffers[stage]])
                if stage==0 and shared_port:available=max(available,ends[1])
                if stage==1 and local_shared_port:available=max(available,ends[2])
                start.append(available);finish.append(available+dt)
            ends=finish
            for stage in range(3):slots[stage][i%buffers[stage]]=finish[stage+1]
            if len(trace)<trace_limit:
                row={'panel':i}
                for stage,name in enumerate(('H','R','S','C')):row.update({name+'_start_s':start[stage],name+'_end_s':finish[stage]})
                trace.append(row)
            i+=1
            if accelerate and i%2==0:
                state=ends+[v for group in slots for v in group]
                if previous is not None:
                    diffs=[a-b for a,b in zip(state,previous)];shift=diffs[0]
                    if shift>0 and all(abs(d-shift)<=max(1e-18,abs(shift)*1e-10) for d in diffs):
                        cycles=(stop-i)//2
                        if cycles and len(trace)>=trace_limit:
                            offset=cycles*shift;ends=[x+offset for x in ends];slots=[[x+offset for x in group] for group in slots]
                            i+=cycles*2;skipped+=cycles*2
                previous=ends+[v for group in slots for v in group]
    return {'duration_s':ends[3],'H_end_s':ends[0],'R_end_s':ends[1],'S_end_s':ends[2],'C_end_s':ends[3],
            'panels':i,'skipped_panels':skipped,'trace':trace,'stages':['HBM','L2','Shared-read','matrix']}
